// Minimal SWG IFF parser for building/object files
// Supports chunk hierarchy and basic data extraction

export interface IffChunk {
  tag: string
  size: number
  offset: number
  children: IffChunk[]
  data?: Uint8Array
}

export function readTag(view: DataView, offset: number): string {
  let tag = ''
  for (let i = 0; i < 4; ++i) tag += String.fromCharCode(view.getUint8(offset + i))
  return tag
}

export function parseIff(buffer: ArrayBuffer): IffChunk {
  const view = new DataView(buffer)

  function isTagLikelyContainer(tag: string): boolean {
    return tag === 'FORM' || tag === 'LIST' || tag === 'CAT '
  }

  function parseWithEndian(littleEndian: boolean): IffChunk {
    function parseChunk(offset: number, end: number): IffChunk {
      if (offset + 8 > end) {
        throw new Error('IFF chunk header out of bounds')
      }

      const tag = readTag(view, offset)
      const size = view.getUint32(offset + 4, littleEndian)
      const chunkStart = offset + 8
      const chunkEnd = chunkStart + size

      if (size < 0 || chunkEnd > end) {
        throw new Error('IFF chunk bounds invalid')
      }

      const children: IffChunk[] = []
      let data: Uint8Array | undefined

      let childOffset = chunkStart
      if (isTagLikelyContainer(tag) && size >= 4) {
        // FORM/LIST/CAT commonly include a 4-byte form type before child chunks.
        childOffset += 4
      }

      while (childOffset + 8 <= chunkEnd) {
        try {
          const childSize = view.getUint32(childOffset + 4, littleEndian)
          if (childSize < 0 || childOffset + 8 + childSize > chunkEnd) break
          children.push(parseChunk(childOffset, chunkEnd))
          childOffset += 8 + childSize
        } catch {
          break
        }
      }

      if (children.length === 0) {
        data = new Uint8Array(buffer, chunkStart, size)
      }

      return { tag, size, offset, children, data }
    }

    return parseChunk(0, buffer.byteLength)
  }

  try {
    return parseWithEndian(true)
  } catch {
    try {
      return parseWithEndian(false)
    } catch {
      // Last-resort fallback to keep editor responsive for variant/broken assets.
      return {
        tag: 'DATA',
        size: buffer.byteLength,
        offset: 0,
        children: [],
        data: new Uint8Array(buffer),
      }
    }
  }
}

function isPrintableAscii(byte: number): boolean {
  return byte >= 32 && byte <= 126
}

export function extractAsciiStrings(buffer: ArrayBufferLike, minLength = 6): string[] {
  const bytes = new Uint8Array(buffer)
  const decoder = new TextDecoder('latin1')
  const strings: string[] = []

  let start = -1
  for (let i = 0; i <= bytes.length; i += 1) {
    const byte = i < bytes.length ? bytes[i] : 0
    if (isPrintableAscii(byte)) {
      if (start === -1) start = i
      continue
    }

    if (start !== -1) {
      const length = i - start
      if (length >= minLength) {
        strings.push(decoder.decode(bytes.subarray(start, i)))
      }
      start = -1
    }
  }

  return strings
}

export interface ExtractedBuildingObject {
  templatePath: string
  x: number
  y: number
  z: number
  yaw: number
}

function findSubarray(bytes: Uint8Array, needle: Uint8Array, from: number): number {
  outer: for (let i = from; i <= bytes.length - needle.length; i += 1) {
    for (let j = 0; j < needle.length; j += 1) {
      if (bytes[i + j] !== needle[j]) continue outer
    }
    return i
  }
  return -1
}

function looksLikeCoordinate(value: number): boolean {
  return Number.isFinite(value) && Math.abs(value) <= 8192
}

function fallbackGridPosition(index: number): { x: number; z: number } {
  const cols = 6
  const row = Math.floor(index / cols)
  const col = index % cols
  return {
    x: (col - (cols - 1) / 2) * 2.8,
    z: (row - 2) * 2.8,
  }
}

function walkIffChunks(root: IffChunk): IffChunk[] {
  const chunks: IffChunk[] = []
  const stack: IffChunk[] = [root]
  while (stack.length > 0) {
    const next = stack.pop()
    if (!next) continue
    chunks.push(next)
    for (const child of next.children) stack.push(child)
  }
  return chunks
}

function readObjectRefsFromData(data: Uint8Array): string[] {
  const refs = new Set<string>()
  const strings = extractAsciiStrings(
    data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength),
    8,
  )

  for (const raw of strings) {
    const normalized = raw.replace(/\\/g, '/').toLowerCase()
    if (!normalized.endsWith('.iff')) continue
    if (normalized.includes('/object/') || normalized.startsWith('object/')) {
      refs.add(normalized)
    }
  }

  return Array.from(refs)
}

function readTransformsFromData(data: Uint8Array): Array<{ x: number; y: number; z: number; yaw: number }> {
  const transforms: Array<{ x: number; y: number; z: number; yaw: number }> = []
  if (data.byteLength < 16) return transforms

  const view = new DataView(data.buffer, data.byteOffset, data.byteLength)
  for (let ptr = 0; ptr + 16 <= data.byteLength; ptr += 4) {
    const x = view.getFloat32(ptr, true)
    const y = view.getFloat32(ptr + 4, true)
    const z = view.getFloat32(ptr + 8, true)
    const yaw = view.getFloat32(ptr + 12, true)
    if (!looksLikeCoordinate(x) || !looksLikeCoordinate(y) || !looksLikeCoordinate(z)) continue
    transforms.push({ x, y, z, yaw: Number.isFinite(yaw) ? yaw : 0 })
    if (transforms.length >= 256) break
  }

  return transforms
}

export function extractBuildingObjectsFromIff(buffer: ArrayBuffer): ExtractedBuildingObject[] {
  // Primary extraction path: pair object template refs and transforms inside the same IFF data chunks.
  try {
    const root = parseIff(buffer)
    const chunks = walkIffChunks(root)
    const structured: ExtractedBuildingObject[] = []

    for (const chunk of chunks) {
      if (!chunk.data || chunk.data.length < 12) continue
      const refs = readObjectRefsFromData(chunk.data)
      if (refs.length === 0) continue

      const transforms = readTransformsFromData(chunk.data)
      refs.forEach((templatePath, index) => {
        const transform = transforms[index] ?? transforms[transforms.length - 1]
        if (transform) {
          structured.push({
            templatePath,
            x: transform.x,
            y: transform.y,
            z: transform.z,
            yaw: transform.yaw,
          })
        } else {
          const fallback = fallbackGridPosition(index)
          structured.push({
            templatePath,
            x: fallback.x,
            y: 0,
            z: fallback.z,
            yaw: 0,
          })
        }
      })
    }

    if (structured.length > 0) return structured
  } catch {
    // Fall back to legacy scan below.
  }

  // Fallback extraction path: broad scan for object refs and nearby transform floats.
  const bytes = new Uint8Array(buffer)
  const encoder = new TextEncoder()
  const view = new DataView(buffer)

  const strings = extractAsciiStrings(buffer, 8)
    .map((s) => s.replace(/\\/g, '/').toLowerCase())
    .filter((s) => s.includes('/') && s.endsWith('.iff'))
    .filter((s) => s.includes('/object/') || s.startsWith('object/'))

  const uniqueTemplates = Array.from(new Set(strings))
  const extracted: ExtractedBuildingObject[] = []

  uniqueTemplates.forEach((templatePath, templateIndex) => {
    const needle = encoder.encode(templatePath)
    let offset = 0
    let occurrences = 0

    while (true) {
      const hit = findSubarray(bytes, needle, offset)
      if (hit < 0) break
      offset = hit + needle.length
      occurrences += 1

      const windowStart = Math.max(0, hit - 96)
      const windowEnd = Math.max(windowStart, hit - 12)

      let best: { x: number; y: number; z: number; yaw: number; distance: number } | null = null

      for (let ptr = windowStart; ptr <= windowEnd; ptr += 4) {
        if (ptr + 16 > bytes.length) break
        const x = view.getFloat32(ptr, true)
        const y = view.getFloat32(ptr + 4, true)
        const z = view.getFloat32(ptr + 8, true)
        const yaw = view.getFloat32(ptr + 12, true)
        if (!looksLikeCoordinate(x) || !looksLikeCoordinate(y) || !looksLikeCoordinate(z)) {
          continue
        }

        const distance = hit - ptr
        if (!best || distance < best.distance) {
          best = { x, y, z, yaw: Number.isFinite(yaw) ? yaw : 0, distance }
        }
      }

      if (best) {
        extracted.push({
          templatePath,
          x: best.x,
          y: best.y,
          z: best.z,
          yaw: best.yaw,
        })
      } else {
        const fallback = fallbackGridPosition(templateIndex + occurrences - 1)
        extracted.push({
          templatePath,
          x: fallback.x,
          y: 0,
          z: fallback.z,
          yaw: 0,
        })
      }
    }
  })

  return extracted
}
