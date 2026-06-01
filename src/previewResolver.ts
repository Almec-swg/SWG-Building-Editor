import { extractAsciiStrings, parseIff, type IffChunk } from './iffParser'

export interface PreviewMeshData {
  positions: number[]
  indices: number[]
  normals?: number[]  // per-vertex normals (XYZ) extracted from VTXA data
  uvs?: number[]   // UV set 0
  uvs1?: number[]  // UV set 1, when FVF texCount >= 2
  uvSets?: number[][] // All declared UV sets by index (0..N-1)
  /** True when the vertex format declares a UV channel; false when confirmed absent; undefined when unknown. */
  hasUvChannel?: boolean
  /** Debug-only: human-readable summary of how the MSH decoder interpreted this part's vertex stream. */
  decodeDebug?: string
}

export interface RepositorySourceFile {
  file: File
  sourceLabel: string
}

export interface ResolvedTemplateVisual {
  templatePath: string
  objectPath?: string
  appearancePath?: string
  shaderPath?: string
  texturePath?: string
  normalTexturePath?: string
  textureAddressU?: TextureAddressMode
  textureAddressV?: TextureAddressMode
  textureMipmapFilter?: TextureFilterMode
  textureMinificationFilter?: TextureFilterMode
  textureMagnificationFilter?: TextureFilterMode
  normalTextureAddressU?: TextureAddressMode
  normalTextureAddressV?: TextureAddressMode
  meshPath?: string
  sourceLabel?: string
  textureSourceLabel?: string
  normalTextureSourceLabel?: string
  mesh?: PreviewMeshData
  meshParts?: PreviewMeshData[]
  meshPartTexturePaths?: string[]
  meshPartNormalTexturePaths?: (string | undefined)[]
  meshPartNormalTextureAddressU?: (TextureAddressMode | undefined)[]
  meshPartNormalTextureAddressV?: (TextureAddressMode | undefined)[]
  meshPartPrimaryUvSetIndices?: number[]
  meshPartPrimaryScaleU?: (number | undefined)[]
  meshPartPrimaryScaleV?: (number | undefined)[]
  meshPartTextureAddressU?: (TextureAddressMode | undefined)[]
  meshPartTextureAddressV?: (TextureAddressMode | undefined)[]
  meshPartTextureMipmapFilter?: (TextureFilterMode | undefined)[]
  meshPartTextureMinificationFilter?: (TextureFilterMode | undefined)[]
  meshPartTextureMagnificationFilter?: (TextureFilterMode | undefined)[]
  meshPartSecondaryTexturePaths?: (string | undefined)[]
  meshPartSecondaryUvSetIndices?: number[]
  meshPartSecondaryTextureAddressU?: (TextureAddressMode | undefined)[]
  meshPartSecondaryTextureAddressV?: (TextureAddressMode | undefined)[]
  meshPartChunkTrace?: (string | undefined)[]
  meshPartShaderPaths?: (string | undefined)[]
  meshPartEffectPaths?: (string | undefined)[]
  meshPartDomains?: ('exterior' | 'interior')[]
  /** Hardpoints (HPNT chunks) discovered along the appearance / cell / CMP chain. */
  hardpoints?: MeshHardpoint[]
  effectPath?: string
  effectOptionCodes?: string[]
  vertexProgramPaths?: string[]
  pixelProgramPaths?: string[]
  shaderStageTexturePaths?: string[]
  shaderStageNormalTexturePaths?: string[]
  status: 'mesh' | 'appearance-only' | 'object-only' | 'missing'
}

/**
 * A SWG HPNT chunk: a named anchor on a mesh/appearance used by the engine to
 * attach child objects (door handles, signage, weapons on racks, etc.) or to
 * mark positions of interest (door portals, FX origins). The matrix is a
 * mesh-local affine transform laid out as three rows of (Rx, Ry, Rz, Tx).
 */
export interface MeshHardpoint {
  name: string
  /** 12 floats, row-major 3x4: [R0xyz, T0, R1xyz, T1, R2xyz, T2]. */
  matrix: number[]
  /** IFF file the hardpoint was discovered in (MSH/CMP/POB/etc.). */
  sourcePath?: string
}

export type RepositoryLookup = (path: string) => Promise<RepositorySourceFile | null>
export interface ResolveTemplateOptions {
  strictDeclaredOnly?: boolean
  preferDecodedOverComposite?: boolean
}

export type TextureAddressMode = 'wrap' | 'mirror' | 'clamp' | 'border' | 'mirroronce'
export type TextureFilterMode = 'none' | 'point' | 'linear' | 'anisotropic' | 'flatcubic' | 'gaussiancubic'

type ShaderResolutionResult = {
  texturePath: string
  textureSourceLabel: string
  textureAddressU?: TextureAddressMode
  textureAddressV?: TextureAddressMode
  textureMipmapFilter?: TextureFilterMode
  textureMinificationFilter?: TextureFilterMode
  textureMagnificationFilter?: TextureFilterMode
  primaryScaleU?: number
  primaryScaleV?: number
  normalTexturePath?: string
  normalTextureSourceLabel?: string
  normalTextureAddressU?: TextureAddressMode
  normalTextureAddressV?: TextureAddressMode
  primaryUvSetIndex: number
  secondaryTexturePath?: string
  secondaryUvSetIndex: number
  secondaryTextureAddressU?: TextureAddressMode
  secondaryTextureAddressV?: TextureAddressMode
  secondaryScaleU?: number
  secondaryScaleV?: number
} | null

const shaderTextureResolutionCache = new Map<string, Promise<ShaderResolutionResult>>()
const shaderStageRefsCache = new Map<string, {
  stageTextures: string[]
  stageNormals: string[]
  hasTxms: boolean
  primaryUvSetIndex: number
  tcssPayloadsParsed: number
  tcssRecognizedAssignments: number
  tcssUniqueAssignments: number
  primaryAddressU?: TextureAddressMode
  primaryAddressV?: TextureAddressMode
  primaryMipmapFilter?: TextureFilterMode
  primaryMinificationFilter?: TextureFilterMode
  primaryMagnificationFilter?: TextureFilterMode
  secondaryTextures: string[]
  secondaryUvSetIndex: number
  secondaryAddressU?: TextureAddressMode
  secondaryAddressV?: TextureAddressMode
  normalAddressU?: TextureAddressMode
  normalAddressV?: TextureAddressMode
  alphaTest: boolean
  alphaReference: number
  transparent: boolean
  alphaBlend: boolean
  effectTags: string[]
  shaderDebugChunks?: string[]
}>()
const effectRecipeCache = new Map<string, {
  optionCodes: string[]
  vertexPrograms: string[]
  pixelPrograms: string[]
}>()
const effectDeclaredTextureRefsCache = new Map<string, Array<{ path: string; slot: ShaderTextureSlot; order: number }>>()
const effectDebugChunksCache = new Map<string, string[]>()
const resolvedRefCandidatesCache = new Map<string, string[]>()

export function clearPreviewResolverCaches(): void {
  shaderTextureResolutionCache.clear()
  shaderStageRefsCache.clear()
  effectRecipeCache.clear()
  effectDeclaredTextureRefsCache.clear()
  effectDebugChunksCache.clear()
  resolvedRefCandidatesCache.clear()
}

export function getShaderDebugChunks(shaderPath: string): string[] | undefined {
  const normalized = normalizeSwgPath(shaderPath)
  const cached = shaderStageRefsCache.get(normalized)
  return cached?.shaderDebugChunks
}

export function getEffectDebugChunks(effectPath: string): string[] | undefined {
  const normalized = normalizeSwgPath(effectPath)
  return effectDebugChunksCache.get(normalized)
}

function captureEffectDebugChunks(effectBuffer: ArrayBuffer, effectPath: string): string[] {
  const normalized = normalizeSwgPath(effectPath)
  const cached = effectDebugChunksCache.get(normalized)
  if (cached) return cached
  const bytes = new Uint8Array(effectBuffer)
  const root = parseBigEndianChunks(bytes)
  const out: string[] = []
  if (!root) {
    effectDebugChunksCache.set(normalized, out)
    return out
  }
  const walk = (node: BinaryChunk, depth = 0) => {
    const indent = '  '.repeat(depth)
    if (node.type) {
      out.push(`${indent}${node.tag}:${node.type}`)
    } else {
      const dataSize = node.dataEnd - node.dataStart
      const maxBytesToShow = 4096
      if (dataSize > 0 && dataSize <= maxBytesToShow) {
        const hex: string[] = []
        const ascii: string[] = []
        for (let i = node.dataStart; i < node.dataEnd; i++) {
          const b = bytes[i]
          hex.push(b.toString(16).padStart(2, '0'))
          ascii.push(b >= 0x20 && b <= 0x7e ? String.fromCharCode(b) : '.')
        }
        out.push(`${indent}${node.tag} (${dataSize}B) [${hex.join(' ')}] "${ascii.join('')}"`)
      } else {
        out.push(`${indent}${node.tag} (${dataSize} bytes - truncated)`)
      }
    }
    for (const child of node.children) walk(child, depth + 1)
  }
  walk(root)
  effectDebugChunksCache.set(normalized, out)
  return out
}

export function getShaderRenderProps(shaderPath: string): {
  alphaTest: boolean
  alphaReference: number
  transparent: boolean
  alphaBlend: boolean
  effectTags: string[]
} | undefined {
  const normalized = normalizeSwgPath(shaderPath)
  const cached = shaderStageRefsCache.get(normalized)
  if (!cached) return undefined
  return {
    alphaTest: cached.alphaTest,
    alphaReference: cached.alphaReference,
    transparent: cached.transparent,
    alphaBlend: cached.alphaBlend,
    effectTags: [...cached.effectTags],
  }
}

const refRegex = /[a-z0-9_./\\-]+\.(?:iff|pob|apt|lod|msh|cmp|flr|sat|sht|eft|vsh|psh|dds|tga)/gi

function ext(path: string): string {
  const dot = path.lastIndexOf('.')
  if (dot < 0) return ''
  return path.slice(dot).toLowerCase()
}

export function normalizeSwgPath(path: string): string {
  return path.replace(/\\/g, '/').replace(/^repository\//i, '').replace(/^\/+/, '').toLowerCase()
}

function dirname(path: string): string {
  const slash = path.lastIndexOf('/')
  if (slash < 0) return ''
  return path.slice(0, slash)
}

function extensionDirectories(fileExt: string): string[] {
  if (fileExt === '.pob') {
    return ['appearance']
  }
  if (fileExt === '.apt' || fileExt === '.sat' || fileExt === '.sht' || fileExt === '.trt') {
    return ['appearance', 'shader', 'effect']
  }
  if (fileExt === '.eft' || fileExt === '.vsh' || fileExt === '.psh') {
    return ['effect', 'vertex_program', 'pixel_program', 'shader']
  }
  if (fileExt === '.msh' || fileExt === '.lod') {
    return ['appearance/mesh', 'appearance', 'mesh']
  }
  if (fileExt === '.cmp') {
    return ['appearance/component', 'appearance', 'component']
  }
  if (fileExt === '.flr') {
    return ['appearance/collision', 'appearance', 'collision']
  }
  if (fileExt === '.dds' || fileExt === '.tga') {
    return ['texture', 'appearance/texture', 'shader/texture']
  }
  if (fileExt === '.iff') {
    return ['object', 'appearance', 'building', 'clientdata']
  }
  return []
}

function resolveRefCandidates(basePath: string, ref: string): string[] {
  const clean = normalizeSwgPath(ref)
  // SWG uses exact paths as they appear in the data files.
  // If the reference contains a directory separator, use it exactly.
  if (clean.includes('/')) return [clean]

  const results: string[] = []
  const seen = new Set<string>()
  const push = (value: string) => {
    const normalized = normalizeSwgPath(value)
    if (!normalized || seen.has(normalized)) return
    seen.add(normalized)
    results.push(normalized)
  }

  // Priority 1: Relative to the referencing file's directory (same directory as shader/mesh)
  const dir = dirname(normalizeSwgPath(basePath))
  if (dir) push(`${dir}/${clean}`)

  // Priority 2: Standard extension-specific directories (texture/, appearance/texture/, etc.)
  const fileExt = ext(clean)
  for (const root of extensionDirectories(fileExt)) {
    push(`${root}/${clean}`)
  }

  // Priority 3: Root level (least preferred - avoid ambiguity)
  push(clean)

  return results
}

function collectRefs(buffer: ArrayBuffer): string[] {
  const refs = new Set<string>()
  const strings = extractAsciiStrings(buffer, 4)

  for (const value of strings) {
    const matches = value.match(refRegex)
    if (!matches) continue
    for (const match of matches) {
      refs.add(normalizeSwgPath(match))
    }
  }

  return Array.from(refs)
}

function getResolvedRefCandidatesCached(basePath: string, buffer: ArrayBuffer): string[] {
  const key = normalizeSwgPath(basePath)
  const cached = resolvedRefCandidatesCache.get(key)
  if (cached) return cached

  const refs = Array.from(
    new Set(collectRefs(buffer).flatMap((r) => resolveRefCandidates(basePath, r))),
  )
  resolvedRefCandidatesCache.set(key, refs)
  return refs
}

function isLikelyNonDiffuseTexturePath(value: string): boolean {
  const file = (value.split('/').pop() ?? value).toLowerCase()
  if (/(_n\.(dds|tga))$/i.test(file)) return true
  if (/(^|[_-])(normal|bump)([_.-]|$)/i.test(file)) return true
  if (/(^|[_-])(env|envmap|environment|cube|cubemap|reflection)([_.-]|$)/i.test(file)) return true
  if (/(^|[_-])(back|detail)([_.-]|$)/i.test(file)) return true
  return false
}

async function collectCmpMeshRefs(
  startPath: string,
  lookup: RepositoryLookup,
  visited = new Set<string>(),
): Promise<string[]> {
  const refs = new Set<string>()
  const norm = normalizeSwgPath(startPath)
  if (visited.has(norm)) return []
  visited.add(norm)
  if (ext(norm) !== '.cmp') return []

  const hit = await lookup(norm)
  if (!hit) return []

  try {
    const buffer = await hit.file.arrayBuffer()
    const candidates = getResolvedRefCandidatesCached(norm, buffer)
    for (const candidate of candidates) {
      refs.add(candidate)
      if (ext(candidate) === '.cmp') {
        const nested = await collectCmpMeshRefs(candidate, lookup, visited)
        for (const nestedRef of nested) refs.add(nestedRef)
      }
    }
  } catch {
    // Ignore malformed or unsupported CMP payloads.
  }

  return Array.from(refs)
}

function buildLikelyComponentCmpCandidates(appearancePath: string): string[] {
  const normalized = normalizeSwgPath(appearancePath)
  const slash = normalized.lastIndexOf('/')
  const fileName = slash >= 0 ? normalized.slice(slash + 1) : normalized
  const stem = fileName.replace(/\.[^.]+$/, '')
  const stemNoL = stem.replace(/_l\d+$/, '')
  const stemNoR = stemNoL.replace(/_r\d+(?:_mesh)?$/, '')

  const stems = Array.from(new Set([stem, stemNoL, stemNoR].filter(Boolean)))
  const out: string[] = []
  const seen = new Set<string>()
  const push = (value: string) => {
    const next = normalizeSwgPath(value)
    if (!next.endsWith('.cmp') || seen.has(next)) return
    seen.add(next)
    out.push(next)
  }

  for (const s of stems) {
    push(`appearance/component/${s}.cmp`)
    push(`appearance/component/${s}_l0.cmp`)
    push(`appearance/component/${s}_r0_mesh_l0.cmp`)
  }

  return out
}

function walkChunks(root: IffChunk): IffChunk[] {
  const out: IffChunk[] = []
  const stack: IffChunk[] = [root]
  while (stack.length > 0) {
    const next = stack.pop()
    if (!next) continue
    out.push(next)
    for (const child of next.children) stack.push(child)
  }
  return out
}

function isLikelyVertex(value: number): boolean {
  return Number.isFinite(value) && Math.abs(value) < 32000
}

interface BinaryChunk {
  tag: string
  start: number
  size: number
  dataStart: number
  dataEnd: number
  type?: string
  children: BinaryChunk[]
}

function readTagRaw(bytes: Uint8Array, offset: number): string {
  return String.fromCharCode(
    bytes[offset],
    bytes[offset + 1],
    bytes[offset + 2],
    bytes[offset + 3],
  )
}

function parseBigEndianChunks(bytes: Uint8Array): BinaryChunk | null {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)

  function parseAt(start: number, end: number): BinaryChunk | null {
    if (start + 8 > end) return null
    const tag = readTagRaw(bytes, start)
    const size = view.getUint32(start + 4, false)
    const dataStart = start + 8
    const dataEnd = dataStart + size
    if (size > bytes.length || dataEnd > end) return null

    const chunk: BinaryChunk = {
      tag,
      start,
      size,
      dataStart,
      dataEnd,
      children: [],
    }

    let cursor = dataStart
    if ((tag === 'FORM' || tag === 'LIST' || tag === 'CAT ') && size >= 4) {
      chunk.type = readTagRaw(bytes, dataStart)
      cursor = dataStart + 4
    }

    while (cursor + 8 <= dataEnd) {
      const childTag = readTagRaw(bytes, cursor)
      if (!/^[\x20-\x7E]{4}$/.test(childTag)) break
      const childSize = view.getUint32(cursor + 4, false)
      if (childSize > bytes.length || cursor + 8 + childSize > dataEnd) break

      const child = parseAt(cursor, dataEnd)
      if (!child) break
      chunk.children.push(child)
      cursor += 8 + childSize
    }

    return chunk
  }

  return parseAt(0, bytes.length)
}

function flattenChunks(root: BinaryChunk): BinaryChunk[] {
  const out: BinaryChunk[] = []
  const stack: BinaryChunk[] = [root]
  while (stack.length > 0) {
    const next = stack.pop()
    if (!next) continue
    out.push(next)
    for (const child of next.children) stack.push(child)
  }
  return out
}

function flattenChunkSubtree(root: BinaryChunk): BinaryChunk[] {
  const out: BinaryChunk[] = []
  const stack: BinaryChunk[] = [...root.children]
  while (stack.length > 0) {
    const next = stack.pop()
    if (!next) continue
    out.push(next)
    for (const child of next.children) stack.push(child)
  }
  return out
}

function decodeVertexTriples(payload: Uint8Array): number[] {
  if (payload.length < 12) return []
  const view = new DataView(payload.buffer, payload.byteOffset, payload.byteLength)

  interface Candidate {
    positions: number[]
    valid: number
    total: number
    score: number
  }

  const decode = (offset: number, littleEndian: boolean, limitCount?: number): Candidate => {
    if (offset < 0 || offset >= payload.length) {
      return { positions: [], valid: 0, total: 0, score: Number.NEGATIVE_INFINITY }
    }

    const maxCount = Math.floor((payload.length - offset) / 12)
    const total = Math.max(0, Math.min(maxCount, limitCount ?? maxCount))
    if (total < 1) {
      return { positions: [], valid: 0, total: 0, score: Number.NEGATIVE_INFINITY }
    }

    const positions: number[] = []
    let valid = 0
    for (let i = 0; i < total; i += 1) {
      const base = offset + i * 12
      const x = view.getFloat32(base, littleEndian)
      const y = view.getFloat32(base + 4, littleEndian)
      const z = view.getFloat32(base + 8, littleEndian)
      if (!isLikelyVertex(x) || !isLikelyVertex(y) || !isLikelyVertex(z)) continue
      positions.push(x, y, z)
      valid += 1
    }

    const density = total > 0 ? valid / total : 0
    const score = valid * 1000 + density * 100
    return { positions, valid, total, score }
  }

  const candidates: Candidate[] = []

  // Baseline decodes: raw chunk interpreted directly as triples.
  candidates.push(decode(0, true))
  candidates.push(decode(0, false))

  // SWG variants often store a count prefix before float triples.
  if (payload.length >= 16) {
    const countBE = view.getUint32(0, false)
    const countLE = view.getUint32(0, true)
    const maxByBytes = Math.floor((payload.length - 4) / 12)

    if (countBE > 0 && countBE <= maxByBytes + 2) {
      candidates.push(decode(4, true, countBE))
      candidates.push(decode(4, false, countBE))
    }
    if (countLE > 0 && countLE <= maxByBytes + 2) {
      candidates.push(decode(4, true, countLE))
      candidates.push(decode(4, false, countLE))
    }
  }

  let best = candidates[0] ?? { positions: [], valid: 0, total: 0, score: Number.NEGATIVE_INFINITY }
  for (const candidate of candidates) {
    if (candidate.score > best.score) best = candidate
  }

  return best.positions
}

function decodeIndices(payload: Uint8Array, vertexCount: number, positions?: number[]): number[] {
  if (payload.length < 6 || vertexCount < 3) return []

  const u16ProbeView = new DataView(payload.buffer, payload.byteOffset, payload.byteLength)
  const u16ProbeCount = Math.min(120, Math.floor(payload.length / 2))
  let oddZero = 0
  let oddTotal = 0
  for (let i = 1; i < u16ProbeCount; i += 2) {
    oddTotal += 1
    if (u16ProbeView.getUint16(i * 2, true) === 0) oddZero += 1
  }
  const looksLikePackedU32 = oddTotal > 0 && oddZero / oddTotal > 0.8

  interface Candidate {
    indices: number[]
    bytesPer: 2 | 4
    stripMode: boolean
    startOffset: number
  }

  const parse = (
    bytesPer: 2 | 4,
    littleEndian: boolean,
    startOffset: number,
    stripMode: boolean,
  ): number[] => {
    if (startOffset >= payload.length) return []
    const usable = payload.subarray(startOffset)
    const view = new DataView(usable.buffer, usable.byteOffset, usable.byteLength)
    const count = Math.floor(usable.length / bytesPer)
    if (count < 3) return []

    const raw: number[] = []
    const separator = bytesPer === 2 ? 0xffff : 0xffffffff
    for (let i = 0; i < count; i += 1) {
      const value = bytesPer === 2
        ? view.getUint16(i * bytesPer, littleEndian)
        : view.getUint32(i * bytesPer, littleEndian)
      if (value === separator || value >= vertexCount) {
        raw.push(-1)
      } else {
        raw.push(value)
      }
    }

    const out: number[] = []
    if (stripMode) {
      for (let i = 2; i < raw.length; i += 1) {
        const a = raw[i - 2]
        const b = raw[i - 1]
        const c = raw[i]
        if (a < 0 || b < 0 || c < 0) continue
        if (a === b || b === c || a === c) continue
        if (i % 2 === 0) out.push(a, b, c)
        else out.push(b, a, c)
      }
    } else {
      for (let i = 0; i + 2 < raw.length; i += 3) {
        const a = raw[i]
        const b = raw[i + 1]
        const c = raw[i + 2]
        if (a < 0 || b < 0 || c < 0) continue
        if (a === b || b === c || a === c) continue
        out.push(a, b, c)
      }
    }

    return out
  }

  const candidates: Candidate[] = []
  for (const bytesPer of [2, 4] as const) {
    for (const littleEndian of [true, false]) {
      const alignment = bytesPer
      const maxOffset = Math.min(64, payload.length - bytesPer * 3)
      for (let offset = 0; offset <= maxOffset; offset += alignment) {
        const list = parse(bytesPer, littleEndian, offset, false)
        if (list.length >= 3) {
          candidates.push({
            indices: list,
            bytesPer,
            stripMode: false,
            startOffset: offset,
          })
        }
        const stripList = parse(bytesPer, littleEndian, offset, true)
        if (stripList.length >= 3) {
          candidates.push({
            indices: stripList,
            bytesPer,
            stripMode: true,
            startOffset: offset,
          })
        }
      }
    }
  }

  if (candidates.length === 0) return []

  const scoreCandidate = (candidate: Candidate): number => {
    const tris = Math.floor(candidate.indices.length / 3)
    if (tris <= 0) return -Infinity

    let penalty = 0
    if (candidate.stripMode) penalty += 180
    if (candidate.startOffset > 0) penalty += candidate.startOffset * 2
    if (looksLikePackedU32 && candidate.bytesPer === 2) penalty += 1200

    let nonMonotonic = 0
    let prev = -1
    const sampleCount = Math.min(1200, candidate.indices.length)
    for (let i = 0; i < sampleCount; i += 1) {
      const value = candidate.indices[i]
      if (value < prev) nonMonotonic += 1
      prev = value
    }
    penalty += nonMonotonic * 0.2

    // Reward candidate length but avoid blindly preferring giant strip outputs.
    return candidate.indices.length - penalty
  }

  if (positions && positions.length >= 9) {
    let bestByTopo: Candidate | null = null
    let bestTopoScore = Number.NEGATIVE_INFINITY
    for (const candidate of candidates) {
      const topoScore = scoreIndexTopology(positions, candidate.indices)
      if (
        !bestByTopo ||
        topoScore > bestTopoScore ||
        (topoScore === bestTopoScore && candidate.indices.length > bestByTopo.indices.length)
      ) {
        bestByTopo = candidate
        bestTopoScore = topoScore
      }
    }
    if (bestByTopo) return bestByTopo.indices
  }

  const best = candidates.sort((a, b) => scoreCandidate(b) - scoreCandidate(a))[0]
  return best.indices
}

function detectPackedU32IndexPattern(payload: Uint8Array): boolean {
  if (payload.length < 16) return false
  const view = new DataView(payload.buffer, payload.byteOffset, payload.byteLength)
  const probeCount = Math.min(120, Math.floor(payload.length / 2))
  let oddZero = 0
  let oddTotal = 0
  for (let i = 1; i < probeCount; i += 2) {
    oddTotal += 1
    if (view.getUint16(i * 2, true) === 0) oddZero += 1
  }
  return oddTotal > 0 && oddZero / oddTotal > 0.8
}

function decodePobIndices(payload: Uint8Array, vertexCount: number): number[] {
  if (payload.length < 6 || vertexCount < 3) return []
  const view = new DataView(payload.buffer, payload.byteOffset, payload.byteLength)
  const preferU32 = detectPackedU32IndexPattern(payload)

  const parseU32List = (offset: number): number[] => {
    const aligned = offset - (offset % 4)
    if (aligned + 12 > payload.length) return []
    const count = Math.floor((payload.length - aligned) / 4)
    const out: number[] = []
    for (let i = 0; i + 2 < count; i += 3) {
      const a = view.getUint32(aligned + i * 4, true)
      const b = view.getUint32(aligned + (i + 1) * 4, true)
      const c = view.getUint32(aligned + (i + 2) * 4, true)
      if (a >= vertexCount || b >= vertexCount || c >= vertexCount) continue
      if (a === b || b === c || a === c) continue
      out.push(a, b, c)
    }
    return out
  }

  const parseU16List = (offset: number): number[] => {
    const aligned = offset - (offset % 2)
    if (aligned + 6 > payload.length) return []
    const count = Math.floor((payload.length - aligned) / 2)
    const out: number[] = []
    for (let i = 0; i + 2 < count; i += 3) {
      const a = view.getUint16(aligned + i * 2, true)
      const b = view.getUint16(aligned + (i + 1) * 2, true)
      const c = view.getUint16(aligned + (i + 2) * 2, true)
      if (a >= vertexCount || b >= vertexCount || c >= vertexCount) continue
      if (a === b || b === c || a === c) continue
      out.push(a, b, c)
    }
    return out
  }

  const u32Candidates: number[][] = []
  for (let offset = 0; offset <= Math.min(32, payload.length - 12); offset += 4) {
    const parsed = parseU32List(offset)
    if (parsed.length >= 3) u32Candidates.push(parsed)
  }

  const u16Candidates: number[][] = []
  for (let offset = 0; offset <= Math.min(32, payload.length - 6); offset += 2) {
    const parsed = parseU16List(offset)
    if (parsed.length >= 3) u16Candidates.push(parsed)
  }

  const bestU32 = u32Candidates.sort((a, b) => b.length - a.length)[0] ?? []
  const bestU16 = u16Candidates.sort((a, b) => b.length - a.length)[0] ?? []

  if (preferU32 && bestU32.length >= 3) return bestU32
  if (bestU32.length >= bestU16.length && bestU32.length >= 3) return bestU32
  if (bestU16.length >= 3) return bestU16
  return []
}

function extractStructuredPobCmeshIdtlMeshes(bytes: Uint8Array, root: BinaryChunk): PreviewMeshData[] {
  const meshes: PreviewMeshData[] = []
  const all = flattenChunks(root)
  const cmeshForms = all.filter((chunk) => chunk.tag === 'FORM' && chunk.type === 'CMSH')

  for (const cmesh of cmeshForms) {
    const descendants = flattenChunkSubtree(cmesh)
    const idtlForms = descendants.filter((chunk) => chunk.tag === 'FORM' && chunk.type === 'IDTL')

    for (const idtl of idtlForms) {
      const idtlDesc = flattenChunkSubtree(idtl)
      const vert = idtlDesc.find((chunk) => chunk.tag === 'VERT' && chunk.size >= 12)
      const indx = idtlDesc.find((chunk) => chunk.tag === 'INDX' && chunk.size >= 12)
      if (!vert || !indx) continue

      const vPayload = bytes.subarray(vert.dataStart, vert.dataEnd)
      const vView = new DataView(vPayload.buffer, vPayload.byteOffset, vPayload.byteLength)
      const vertCount = Math.floor(vPayload.length / 12)
      if (vertCount < 3) continue

      const positions: number[] = []
      for (let i = 0; i < vertCount; i += 1) {
        const base = i * 12
        const x = vView.getFloat32(base, true)
        const y = vView.getFloat32(base + 4, true)
        const z = vView.getFloat32(base + 8, true)
        if (!isLikelyVertex(x) || !isLikelyVertex(y) || !isLikelyVertex(z)) continue
        positions.push(x, y, z)
      }

      const usableVerts = Math.floor(positions.length / 3)
      if (usableVerts < 3) continue

      const iPayload = bytes.subarray(indx.dataStart, indx.dataEnd)
      const iView = new DataView(iPayload.buffer, iPayload.byteOffset, iPayload.byteLength)
      const indices: number[] = []

      // Vanguard/SWG parser reads IDTL indices as int32 triples and flips winding (c,b,a).
      if (iPayload.length % 12 === 0) {
        const triCount = Math.floor(iPayload.length / 12)
        for (let t = 0; t < triCount; t += 1) {
          const base = t * 12
          const a = iView.getInt32(base, true)
          const b = iView.getInt32(base + 4, true)
          const c = iView.getInt32(base + 8, true)
          if (a < 0 || b < 0 || c < 0 || a >= usableVerts || b >= usableVerts || c >= usableVerts) continue
          if (a === b || b === c || a === c) continue
          indices.push(c, b, a)
        }
      }

      if (indices.length < 3) continue
      const mesh = pruneSpikyTriangles({ positions, indices })
      if (!mesh || mesh.indices.length < 3) continue
      meshes.push(mesh)
    }
  }

  return meshes
}

function decodeStructuredMshIndices(payload: Uint8Array, vertexCount: number): number[] {
  if (payload.length < 10 || vertexCount < 3) return []
  const view = new DataView(payload.buffer, payload.byteOffset, payload.byteLength)

  const countLE = view.getInt32(0, true)
  const countBE = view.getInt32(0, false)
  const candidates: number[] = []
  if (countLE > 0) candidates.push(countLE)
  if (countBE > 0 && countBE !== countLE) candidates.push(countBE)

  for (const count of candidates) {
    const byteLen = 4 + count * 2
    if (count < 3 || byteLen > payload.length) continue

    const out: number[] = []
    for (let i = 0; i + 2 < count; i += 3) {
      const a = view.getUint16(4 + i * 2, true)
      const b = view.getUint16(4 + (i + 1) * 2, true)
      const c = view.getUint16(4 + (i + 2) * 2, true)
      if (a >= vertexCount || b >= vertexCount || c >= vertexCount) continue
      if (a === b || b === c || a === c) continue
      out.push(a, b, c)
    }

    if (out.length >= 3) return out
  }

  return []
}

function uvOffsetFromFvfFlags(flags: number, bytesPerVertex: number): number {
  // Determine whether normals (12 bytes) are present between position and UVs.
  // SWG exterior/complex meshes always store pos(12)+norm(12)+uvs.
  // Some interior meshes omit normals: pos(12)+uvs only.
  // We infer this from the stride: if the stride is at least 24 + uvCount*8,
  // normals must be present; if it's only 12 + uvCount*8, they are absent.
  const uvCount = Math.max(1, (flags & 0x0f00) >>> 8)
  const withNormals = 24 + uvCount * 8
  if (bytesPerVertex >= withNormals) return 24
  return 12
}

function extractUvsFromFvfFlags(
  dataPayload: Uint8Array,
  numVertices: number,
  bytesPerVertex: number,
  vertexFlags: number,
  uvSetIndex = 0,
): number[] | undefined {
  // FVF parity:
  //  - For uvSetIndex 0 we accept any bytesPerVertex >= expectedStride. Extra
  //    bytes commonly carry tangent + handedness (trailing) for normal-mapped
  //    meshes, or D3DCOLOR diffuse/specular (between normal and UV) for some
  //    interior meshes. We try the canonical offset first and, if the values
  //    look like sentinels (non-finite or magnitudes far outside any plausible
  //    UV range), we walk forward in 4-byte steps until we find a stream that
  //    passes a sanity check. This rescues palace-exterior (UV at the
  //    canonical offset with trailing tangent) and palace-interior (UV after
  //    a 4-byte color field).
  //  - For uvSetIndex >= 1 we still require exact parity, because guessing a
  //    secondary UV offset is fragile and almost always wrong when the stride
  //    does not match the declared texCount.
  const texCount = (vertexFlags & 0x0f00) >>> 8
  if (texCount === 0 || uvSetIndex >= texCount) return undefined
  const baseUvOffset = uvOffsetFromFvfFlags(vertexFlags, bytesPerVertex)
  const expectedStride = baseUvOffset + texCount * 8
  if (uvSetIndex === 0) {
    if (bytesPerVertex < expectedStride) return undefined
  } else {
    if (expectedStride !== bytesPerVertex) return undefined
  }

  const canonicalOffset = baseUvOffset + uvSetIndex * 8
  if (canonicalOffset + 8 > bytesPerVertex) return undefined

  const view = new DataView(dataPayload.buffer, dataPayload.byteOffset, dataPayload.byteLength)

  // Sanity threshold for "looks like a UV": values must be finite and within
  // a generous tiling envelope. Real SWG UVs occasionally tile beyond [0,1]
  // (e.g. -6..9 on the throne mesh) but never approach 1e6 — that magnitude
  // only appears when we land in a packed-color / NaN-sentinel field.
  const isPlausibleUv = (u: number, v: number): boolean => {
    if (!Number.isFinite(u) || !Number.isFinite(v)) return false
    if (Math.abs(u) > 1_000_000 || Math.abs(v) > 1_000_000) return false
    return true
  }

  // For set 0 we may need to shift forward past padding/color fields. For
  // higher sets we keep the canonical offset only.
  const maxShift = uvSetIndex === 0 ? bytesPerVertex - (canonicalOffset + 8) : 0
  for (let shift = 0; shift <= maxShift; shift += 4) {
    const uvOffset = canonicalOffset + shift
    if (uvOffset + 8 > bytesPerVertex) break
    const out: number[] = []
    let allPlausible = true
    for (let i = 0; i < numVertices; i++) {
      const base = i * bytesPerVertex + uvOffset
      if (base + 8 > dataPayload.length) { allPlausible = false; break }
      const u = view.getFloat32(base, true)
      const v = view.getFloat32(base + 4, true)
      if (!isPlausibleUv(u, v)) { allPlausible = false; break }
      out.push(u, v)
    }
    if (allPlausible && out.length >= numVertices * 2) return out
  }
  return undefined
}

function scoreUvCandidate(uvs: number[] | undefined, numVertices: number): number {
  if (!uvs || uvs.length < numVertices * 2) return Number.NEGATIVE_INFINITY

  let minU = Number.POSITIVE_INFINITY
  let minV = Number.POSITIVE_INFINITY
  let maxU = Number.NEGATIVE_INFINITY
  let maxV = Number.NEGATIVE_INFINITY

  for (let i = 0; i + 1 < numVertices * 2; i += 2) {
    const u = uvs[i]
    const v = uvs[i + 1]
    if (!Number.isFinite(u) || !Number.isFinite(v)) return Number.NEGATIVE_INFINITY
    if (Math.abs(u) > 1_000_000 || Math.abs(v) > 1_000_000) return Number.NEGATIVE_INFINITY
    if (u < minU) minU = u
    if (u > maxU) maxU = u
    if (v < minV) minV = v
    if (v > maxV) maxV = v
  }

  const rangeU = Math.max(0, maxU - minU)
  const rangeV = Math.max(0, maxV - minV)
  if (rangeU < 1e-7 && rangeV < 1e-7) return Number.NEGATIVE_INFINITY
  return rangeU * rangeV + Math.max(rangeU, rangeV) * 0.01
}

function extractStructuredUvsHeuristic(
  dataPayload: Uint8Array,
  numVertices: number,
  bytesPerVertex: number,
): number[] | undefined {
  if (numVertices < 3 || bytesPerVertex < 20) return undefined
  const view = new DataView(dataPayload.buffer, dataPayload.byteOffset, dataPayload.byteLength)

  const candidateOffsets: number[] = []
  const pushOffset = (offset: number) => {
    if (offset < 12 || offset + 8 > bytesPerVertex) return
    if (!candidateOffsets.includes(offset)) candidateOffsets.push(offset)
  }

  // Typical SWG layouts: pos+normal+uv at 24, or pos+uv at 12.
  pushOffset(24)
  pushOffset(12)
  for (let offset = 12; offset + 8 <= bytesPerVertex; offset += 4) {
    pushOffset(offset)
  }

  let best: number[] | undefined
  let bestScore = Number.NEGATIVE_INFINITY

  for (const uvOffset of candidateOffsets) {
    const out: number[] = []
    let valid = true
    for (let i = 0; i < numVertices; i += 1) {
      const base = i * bytesPerVertex + uvOffset
      if (base + 8 > dataPayload.length) {
        valid = false
        break
      }
      const u = view.getFloat32(base, true)
      const v = view.getFloat32(base + 4, true)
      out.push(u, v)
    }
    if (!valid) continue

    const score = scoreUvCandidate(out, numVertices)
    if (score > bestScore) {
      bestScore = score
      best = out
    }
  }

  return Number.isFinite(bestScore) && bestScore > 1e-6 ? best : undefined
}

function extractStructuredUvs(
  dataPayload: Uint8Array,
  numVertices: number,
  bytesPerVertex: number,
  vertexFlags?: number,
): number[] | undefined {
  if (numVertices < 3 || bytesPerVertex < 16) return undefined

  // Data-only mode: decode UVs only from explicit FVF declarations.
  if (vertexFlags !== undefined && vertexFlags !== 0) {
    const strict = extractUvsFromFvfFlags(dataPayload, numVertices, bytesPerVertex, vertexFlags)
    if (Number.isFinite(scoreUvCandidate(strict, numVertices))) {
      return strict
    }
    return extractStructuredUvsHeuristic(dataPayload, numVertices, bytesPerVertex)
  }
  return extractStructuredUvsHeuristic(dataPayload, numVertices, bytesPerVertex)
}

function decodeStructuredMeshFromScope(bytes: Uint8Array, scope: BinaryChunk): PreviewMeshData | null {
  const descendants = flattenChunkSubtree(scope)
  const vtxa = descendants.find((chunk) => chunk.tag === 'FORM' && chunk.type === 'VTXA')
  if (!vtxa) return null

  const vtxaDesc = flattenChunkSubtree(vtxa)
  const info = vtxaDesc.find((chunk) => chunk.tag === 'INFO' && chunk.size >= 8)
  const data = vtxaDesc.find((chunk) => chunk.tag === 'DATA' && chunk.size >= 12)
  if (!info || !data) return null

  const scopeIndices = descendants.filter((chunk) => chunk.tag === 'INDX' && chunk.size >= 10)
  if (scopeIndices.length === 0) return null

  let bestIndx: BinaryChunk | null = null
  let bestDistance = Number.POSITIVE_INFINITY
  for (const indx of scopeIndices) {
    const distance = Math.abs(indx.start - data.start)
    if (distance < bestDistance) {
      bestDistance = distance
      bestIndx = indx
    }
  }
  if (!bestIndx) return null

  const infoView = new DataView(bytes.buffer, bytes.byteOffset + info.dataStart, info.size)
  // INFO[0..3] = DirectX FVF vertex format flags (LE). INFO[4..7] = vertex count.
  const vertexFlags = info.size >= 8 ? infoView.getUint32(0, true) : 0
  const numVerticesLE = infoView.getInt32(4, true)
  const numVerticesBE = infoView.getInt32(4, false)
  const numVertices = numVerticesLE > 0 ? numVerticesLE : numVerticesBE
  if (!numVertices || numVertices < 3) return null

  const bytesPerVertex = Math.floor(data.size / numVertices)
  if (bytesPerVertex < 12 || bytesPerVertex > 128) return null

  const dataPayload = bytes.subarray(data.dataStart, data.dataEnd)
  const dataView = new DataView(dataPayload.buffer, dataPayload.byteOffset, dataPayload.byteLength)
  const positions: number[] = []
  const normals: number[] = []
  // Normals are present when the stride is large enough: pos(12)+norm(12)+uvs.
  // Interior meshes sometimes omit normals (pos+uvs only), detected by the same
  // stride heuristic used for UV offset selection.
  const hasNormalsInVertex = uvOffsetFromFvfFlags(vertexFlags, bytesPerVertex) === 24
  for (let i = 0; i < numVertices; i += 1) {
    const base = i * bytesPerVertex
    if (base + 12 > dataPayload.length) break
    const x = dataView.getFloat32(base, true)
    const y = dataView.getFloat32(base + 4, true)
    const z = dataView.getFloat32(base + 8, true)
    positions.push(
      Number.isFinite(x) ? x : 0,
      Number.isFinite(y) ? y : 0,
      Number.isFinite(z) ? z : 0,
    )
    if (hasNormalsInVertex && base + 24 <= dataPayload.length) {
      const nx = dataView.getFloat32(base + 12, true)
      const ny = dataView.getFloat32(base + 16, true)
      const nz = dataView.getFloat32(base + 20, true)
      normals.push(
        Number.isFinite(nx) ? nx : 0,
        Number.isFinite(ny) ? ny : 0,
        Number.isFinite(nz) ? nz : 0,
      )
    }
  }

  const vertexCount = Math.floor(positions.length / 3)
  if (vertexCount < 3) return null
  const uvs = extractStructuredUvs(dataPayload, vertexCount, bytesPerVertex, vertexFlags)
  // hasUvChannel is authoritative from FVF flags when available; otherwise infer from extraction.
  const fvfTexCount = vertexFlags ? (vertexFlags & 0x0f00) >>> 8 : undefined
  const hasUvChannel = fvfTexCount !== undefined ? fvfTexCount > 0 : uvs !== undefined
  const uvs1 = fvfTexCount !== undefined && fvfTexCount >= 2
    ? extractUvsFromFvfFlags(dataPayload, vertexCount, bytesPerVertex, vertexFlags, 1)
    : undefined

  const iPayload = bytes.subarray(bestIndx.dataStart, bestIndx.dataEnd)
  const strict = decodeStructuredMshIndices(iPayload, vertexCount)
  const fallback = strict.length >= 3 ? strict : decodeIndices(iPayload, vertexCount, positions)
  if (fallback.length < 3) return null

  const meshNormals = normals.length === vertexCount * 3 ? normals : undefined
  const allUvSets = fvfTexCount !== undefined && fvfTexCount > 0
    ? Array.from({ length: fvfTexCount }, (_, uvSetIndex) =>
        extractUvsFromFvfFlags(dataPayload, vertexCount, bytesPerVertex, vertexFlags, uvSetIndex),
      ).filter((set): set is number[] => Boolean(set && set.length >= vertexCount * 2))
    : []

  // Build a decode-debug summary so the surface picker can show what the MSH decoder actually saw.
  const decodeDebugLines: string[] = []
  decodeDebugLines.push(`vertexFlags=0x${vertexFlags.toString(16).padStart(8, '0')} fvfTexCount=${fvfTexCount ?? 'n/a'} bytesPerVertex=${bytesPerVertex} numVertices=${numVertices} hasNormalsInVertex=${hasNormalsInVertex} uvOffset=${uvOffsetFromFvfFlags(vertexFlags, bytesPerVertex)}`)
  const maxBytesToDump = Math.min(dataPayload.length, bytesPerVertex * Math.min(vertexCount, 8))
  const hexBytes: string[] = []
  for (let i = 0; i < maxBytesToDump; i++) hexBytes.push(dataPayload[i].toString(16).padStart(2, '0'))
  for (let v = 0; v < Math.min(vertexCount, 8); v++) {
    const start = v * bytesPerVertex * 2
    const end = start + bytesPerVertex * 2
    const hex = hexBytes.slice(v * bytesPerVertex, (v + 1) * bytesPerVertex).join(' ')
    decodeDebugLines.push(`vtxRaw[${v}] ${hex}`)
    void start; void end
  }
  for (let s = 0; s < allUvSets.length; s++) {
    const set = allUvSets[s]
    let minU = Infinity, maxU = -Infinity, minV = Infinity, maxV = -Infinity
    for (let i = 0; i + 1 < set.length; i += 2) {
      const u = set[i], vv = set[i + 1]
      if (u < minU) minU = u; if (u > maxU) maxU = u
      if (vv < minV) minV = vv; if (vv > maxV) maxV = vv
    }
    decodeDebugLines.push(`uvSet[${s}] range=(${minU.toFixed(4)},${minV.toFixed(4)})->(${maxU.toFixed(4)},${maxV.toFixed(4)})`)
  }
  // Try alternate UV offsets so we can see if there's a stream that actually varies with position.
  for (let altOffset = 12; altOffset + 8 <= bytesPerVertex; altOffset += 4) {
    if (altOffset === uvOffsetFromFvfFlags(vertexFlags, bytesPerVertex)) continue
    let minU = Infinity, maxU = -Infinity, minV = Infinity, maxV = -Infinity
    let allFinite = true
    for (let i = 0; i < vertexCount; i++) {
      const base = i * bytesPerVertex + altOffset
      if (base + 8 > dataPayload.length) { allFinite = false; break }
      const u = dataView.getFloat32(base, true)
      const v = dataView.getFloat32(base + 4, true)
      if (!Number.isFinite(u) || !Number.isFinite(v) || Math.abs(u) > 1e6 || Math.abs(v) > 1e6) { allFinite = false; break }
      if (u < minU) minU = u; if (u > maxU) maxU = u
      if (v < minV) minV = v; if (v > maxV) maxV = v
    }
    if (allFinite) {
      decodeDebugLines.push(`altOffset=${altOffset} range=(${minU.toFixed(4)},${minV.toFixed(4)})->(${maxU.toFixed(4)},${maxV.toFixed(4)})`)
    }
  }
  const decodeDebug = decodeDebugLines.join('\n')

  const raw: PreviewMeshData = {
    positions,
    indices: fallback,
    normals: meshNormals,
    uvs,
    uvs1,
    uvSets: allUvSets,
    hasUvChannel,
    decodeDebug,
  }
  const pruned = pruneSpikyTriangles(raw)
  if (pruned && pruned.indices.length >= 3) {
    pruned.decodeDebug = decodeDebug
    return pruned
  }
  return raw
}

function extractStrictSpsMshMeshes(bytes: Uint8Array, root: BinaryChunk): PreviewMeshData[] {
  const meshes: PreviewMeshData[] = []

  const forms = flattenChunks(root).filter((chunk) => chunk.tag === 'FORM')
  const spsRoot = forms.find((chunk) => chunk.type === 'SPS ')
  if (!spsRoot) return meshes

  const spsVersion = spsRoot.children.find((child) => child.tag === 'FORM')
  if (!spsVersion) return meshes

  const partForms = spsVersion.children.filter(
    (child) => child.tag === 'FORM' && !!child.type && /^\d{4}$/.test(child.type),
  )

  for (const partForm of partForms) {
    const mesh = decodeStructuredMeshFromScope(bytes, partForm)
    if (mesh && mesh.indices.length >= 3) meshes.push(mesh)
  }

  return meshes
}

function extractStructuredMshMeshes(bytes: Uint8Array, root: BinaryChunk): PreviewMeshData[] {
  const strictSpsMeshes = extractStrictSpsMshMeshes(bytes, root)
  if (strictSpsMeshes.length > 0) return strictSpsMeshes

  const meshes: PreviewMeshData[] = []
  interface ScopePair {
    vtxa: BinaryChunk
    info: BinaryChunk
    data: BinaryChunk
    indx: BinaryChunk
  }

  const scopedPairs: ScopePair[] = []

  const visit = (node: BinaryChunk): { hasMeshData: boolean } => {
    let childHasMeshData = false
    for (const child of node.children) {
      const childState = visit(child)
      if (childState.hasMeshData) childHasMeshData = true
    }

    const descendants = flattenChunkSubtree(node)
    const vtxaForms = descendants.filter((chunk) => chunk.tag === 'FORM' && chunk.type === 'VTXA')
    const indxChunks = descendants.filter((chunk) => chunk.tag === 'INDX' && chunk.size >= 10)
    const hasMeshData = vtxaForms.length > 0 && indxChunks.length > 0

    if (hasMeshData && !childHasMeshData) {
      const used = new Set<number>()
      for (const vtxa of vtxaForms) {
        const vtxaDesc = flattenChunkSubtree(vtxa)
        const info = vtxaDesc.find((chunk) => chunk.tag === 'INFO' && chunk.size >= 8)
        const data = vtxaDesc.find((chunk) => chunk.tag === 'DATA' && chunk.size >= 12)
        if (!info || !data) continue

        let bestIdx = -1
        let bestDistance = Number.POSITIVE_INFINITY
        for (let i = 0; i < indxChunks.length; i += 1) {
          if (used.has(i)) continue
          const distance = Math.abs(indxChunks[i].start - data.start)
          if (distance < bestDistance) {
            bestDistance = distance
            bestIdx = i
          }
        }

        if (bestIdx >= 0) {
          used.add(bestIdx)
          scopedPairs.push({
            vtxa,
            info,
            data,
            indx: indxChunks[bestIdx],
          })
        }
      }
    }

    return { hasMeshData }
  }

  visit(root)

  for (const pair of scopedPairs) {
    const { info, data, indx } = pair

    const infoView = new DataView(bytes.buffer, bytes.byteOffset + info.dataStart, info.size)
    // INFO[0..3] = DirectX FVF vertex format flags (LE). INFO[4..7] = vertex count.
    const vertexFlags = info.size >= 8 ? infoView.getUint32(0, true) : 0
    const numVerticesLE = infoView.getInt32(4, true)
    const numVerticesBE = infoView.getInt32(4, false)
    const numVertices = numVerticesLE > 0 ? numVerticesLE : numVerticesBE
    if (!numVertices || numVertices < 3) continue

    const bytesPerVertex = Math.floor(data.size / numVertices)
    if (bytesPerVertex < 12 || bytesPerVertex > 128) continue

    const dataPayload = bytes.subarray(data.dataStart, data.dataEnd)
    const dataView = new DataView(dataPayload.buffer, dataPayload.byteOffset, dataPayload.byteLength)
    const positions: number[] = []
    const normals: number[] = []
    const hasNormalsInVertex = bytesPerVertex >= 24

    for (let i = 0; i < numVertices; i += 1) {
      const base = i * bytesPerVertex
      if (base + 12 > dataPayload.length) break
      const x = dataView.getFloat32(base, true)
      const y = dataView.getFloat32(base + 4, true)
      const z = dataView.getFloat32(base + 8, true)
      positions.push(
        Number.isFinite(x) ? x : 0,
        Number.isFinite(y) ? y : 0,
        Number.isFinite(z) ? z : 0,
      )
      if (hasNormalsInVertex && base + 24 <= dataPayload.length) {
        const nx = dataView.getFloat32(base + 12, true)
        const ny = dataView.getFloat32(base + 16, true)
        const nz = dataView.getFloat32(base + 20, true)
        normals.push(
          Number.isFinite(nx) ? nx : 0,
          Number.isFinite(ny) ? ny : 0,
          Number.isFinite(nz) ? nz : 0,
        )
      }
    }

    const vertexCount = Math.floor(positions.length / 3)
    if (vertexCount < 3) continue
    const uvs = extractStructuredUvs(dataPayload, vertexCount, bytesPerVertex, vertexFlags)
    const fvfTexCount = vertexFlags ? (vertexFlags & 0x0f00) >>> 8 : undefined
    const hasUvChannel = fvfTexCount !== undefined ? fvfTexCount > 0 : uvs !== undefined
    const uvSets = fvfTexCount !== undefined && fvfTexCount > 0
      ? Array.from({ length: fvfTexCount }, (_, uvSetIndex) =>
        extractUvsFromFvfFlags(dataPayload, vertexCount, bytesPerVertex, vertexFlags, uvSetIndex),
      ).filter((set): set is number[] => Boolean(set && set.length >= vertexCount * 2))
      : []
    const uvs1 = fvfTexCount !== undefined && fvfTexCount >= 2
      ? extractUvsFromFvfFlags(dataPayload, vertexCount, bytesPerVertex, vertexFlags, 1)
      : undefined
    const meshNormals = normals.length === vertexCount * 3 ? normals : undefined

    let best: PreviewMeshData | null = null
    let bestScore = Number.NEGATIVE_INFINITY

    const iPayload = bytes.subarray(indx.dataStart, indx.dataEnd)
    const strict = decodeStructuredMshIndices(iPayload, vertexCount)
    const fallback = strict.length >= 3 ? strict : decodeIndices(iPayload, vertexCount, positions)
    if (fallback.length < 3) continue

    const raw: PreviewMeshData = {
      positions,
      indices: fallback,
      normals: meshNormals,
      uvs,
      uvs1,
      uvSets: uvSets.length > 0 ? uvSets : undefined,
      hasUvChannel,
    }
    // DEBUG: capture how the strict SPS decoder interpreted this part's vertex stream.
    {
      const ddLines: string[] = []
      ddLines.push(`[strictSps] vertexFlags=0x${vertexFlags.toString(16).padStart(8, '0')} fvfTexCount=${fvfTexCount ?? 'n/a'} bytesPerVertex=${bytesPerVertex} numVertices=${numVertices} hasNormalsInVertex=${hasNormalsInVertex}`)
      const dumpVerts = Math.min(vertexCount, 8)
      for (let v = 0; v < dumpVerts; v++) {
        const start = v * bytesPerVertex
        const end = Math.min(start + bytesPerVertex, dataPayload.length)
        const hex: string[] = []
        for (let b = start; b < end; b++) hex.push(dataPayload[b].toString(16).padStart(2, '0'))
        ddLines.push(`vtxRaw[${v}] ${hex.join(' ')}`)
      }
      for (let s = 0; s < uvSets.length; s++) {
        const set = uvSets[s]
        let minU = Infinity, maxU = -Infinity, minV = Infinity, maxV = -Infinity
        for (let i = 0; i + 1 < set.length; i += 2) {
          const u = set[i], vv = set[i + 1]
          if (u < minU) minU = u; if (u > maxU) maxU = u
          if (vv < minV) minV = vv; if (vv > maxV) maxV = vv
        }
        ddLines.push(`uvSet[${s}] range=(${minU.toFixed(4)},${minV.toFixed(4)})->(${maxU.toFixed(4)},${maxV.toFixed(4)})`)
      }
      for (let altOffset = 12; altOffset + 8 <= bytesPerVertex; altOffset += 4) {
        let minU = Infinity, maxU = -Infinity, minV = Infinity, maxV = -Infinity
        let ok = true
        for (let i = 0; i < vertexCount; i++) {
          const base2 = i * bytesPerVertex + altOffset
          if (base2 + 8 > dataPayload.length) { ok = false; break }
          const u = dataView.getFloat32(base2, true)
          const v2 = dataView.getFloat32(base2 + 4, true)
          if (!Number.isFinite(u) || !Number.isFinite(v2) || Math.abs(u) > 1e6 || Math.abs(v2) > 1e6) { ok = false; break }
          if (u < minU) minU = u; if (u > maxU) maxU = u
          if (v2 < minV) minV = v2; if (v2 > maxV) maxV = v2
        }
        if (ok) ddLines.push(`altOffset=${altOffset} range=(${minU.toFixed(4)},${minV.toFixed(4)})->(${maxU.toFixed(4)},${maxV.toFixed(4)})`)
      }
      raw.decodeDebug = ddLines.join('\n')
    }
    const rawScore = scoreIndexTopology(raw.positions, raw.indices)
    if (!Number.isFinite(rawScore)) continue

    const pruned = pruneSpikyTriangles(raw)
    const prunedScore = pruned && pruned.indices.length >= 3
      ? scoreIndexTopology(pruned.positions, pruned.indices)
      : Number.NEGATIVE_INFINITY

    const candidate = Number.isFinite(prunedScore) && prunedScore > rawScore ? pruned as PreviewMeshData : raw
    const candidateScore = Number.isFinite(prunedScore) && prunedScore > rawScore ? prunedScore : rawScore

    if (!best || candidateScore > bestScore) {
      best = candidate
      bestScore = candidateScore
    }

    if (best) meshes.push(best)
  }

  return meshes
}

function collectScopedVertIndexPairs(root: BinaryChunk): Array<{ vert: BinaryChunk; indx: BinaryChunk }> {
  const out: Array<{ vert: BinaryChunk; indx: BinaryChunk }> = []
  const stack: BinaryChunk[] = [root]

  while (stack.length > 0) {
    const node = stack.pop()
    if (!node) continue

    const verts = node.children.filter((child) => child.tag === 'VERT' && child.size >= 12)
    const indices = node.children.filter((child) => child.tag === 'INDX' && child.size >= 6)

    if (verts.length > 0 && indices.length > 0) {
      const used = new Set<number>()
      for (const vert of verts) {
        let bestIdx = -1
        let bestDistance = Number.POSITIVE_INFINITY
        for (let i = 0; i < indices.length; i += 1) {
          if (used.has(i)) continue
          const distance = Math.abs(indices[i].start - vert.start)
          if (distance < bestDistance) {
            bestDistance = distance
            bestIdx = i
          }
        }
        if (bestIdx >= 0) {
          used.add(bestIdx)
          out.push({ vert, indx: indices[bestIdx] })
        }
      }
    }

    for (const child of node.children) stack.push(child)
  }

  return out
}

function extractMeshFromPob(buffer: ArrayBuffer): PreviewMeshData | null {
  const bytes = new Uint8Array(buffer)
  const root = parseBigEndianChunks(bytes)
  if (!root) return null

  const structuredMeshes = extractStructuredPobCmeshIdtlMeshes(bytes, root)
  if (structuredMeshes.length > 0) {
    return combineMeshes(structuredMeshes)
  }

  const meshes: PreviewMeshData[] = []

  const scopedPairs = collectScopedVertIndexPairs(root)

  for (const pair of scopedPairs) {
    const vPayload = bytes.subarray(pair.vert.dataStart, pair.vert.dataEnd)
    const positions = decodeVertexTriples(vPayload)
    const vertexCount = Math.floor(positions.length / 3)
    if (vertexCount < 3) continue

    const iPayload = bytes.subarray(pair.indx.dataStart, pair.indx.dataEnd)
    const candidates = [
      decodePobIndices(iPayload, vertexCount),
      decodeIndices(iPayload, vertexCount, positions),
    ].filter((list) => list.length >= 3)
    if (candidates.length === 0) continue

    let bestIndices: number[] | null = null
    let bestScore = Number.NEGATIVE_INFINITY
    for (const candidate of candidates) {
      const score = scoreIndexTopology(positions, candidate)
      if (!bestIndices || score > bestScore) {
        bestIndices = candidate
        bestScore = score
      }
    }

    if (!bestIndices) continue
    const pruned = pruneSpikyTriangles({ positions, indices: bestIndices })
    if (!pruned || pruned.indices.length < 3) continue
    meshes.push(pruned)
  }

  if (meshes.length > 0) {
    return combineMeshes(meshes)
  }

  // Fallback only when no scoped pairs are available.
  const flat = flattenChunks(root)
  const vertChunks = flat.filter((chunk) => chunk.tag === 'VERT' && chunk.size >= 12)
  const indexChunks = flat.filter((chunk) => chunk.tag === 'INDX' && chunk.size >= 6)
  if (vertChunks.length === 0 || indexChunks.length === 0) return null

  for (let vi = 0; vi < vertChunks.length; vi += 1) {
    const v = vertChunks[vi]
    const vPayload = bytes.subarray(v.dataStart, v.dataEnd)
    const positions = decodeVertexTriples(vPayload)
    const vertexCount = Math.floor(positions.length / 3)
    if (vertexCount < 3) continue

    let bestMesh: PreviewMeshData | null = null
    let bestScore = Number.NEGATIVE_INFINITY

    for (let ii = 0; ii < indexChunks.length; ii += 1) {
      const i = indexChunks[ii]
      const iPayload = bytes.subarray(i.dataStart, i.dataEnd)
      const pobDecoded = decodePobIndices(iPayload, vertexCount)
      const genericDecoded = decodeIndices(iPayload, vertexCount, positions)
      const candidates = [pobDecoded, genericDecoded].filter((list) => list.length >= 3)
      if (candidates.length === 0) continue

      for (const indices of candidates) {
        const rawMesh: PreviewMeshData = { positions, indices }
        const rawTopology = scoreIndexTopology(rawMesh.positions, rawMesh.indices)
        if (!Number.isFinite(rawTopology)) continue

        let candidateMesh = rawMesh
        let candidateTopology = rawTopology

        const pruned = pruneSpikyTriangles(rawMesh)
        if (pruned && pruned.indices.length >= 3) {
          const prunedTopology = scoreIndexTopology(pruned.positions, pruned.indices)
          if (Number.isFinite(prunedTopology) && prunedTopology > rawTopology) {
            candidateMesh = pruned
            candidateTopology = prunedTopology
          }
        }

        const chunkDistance = Math.abs(i.dataStart - v.dataStart)
        const proximityPenalty = chunkDistance * 0.02
        const score = candidateTopology - proximityPenalty

        if (!bestMesh || score > bestScore) {
          bestMesh = candidateMesh
          bestScore = score
        }
      }
    }

    if (bestMesh && bestScore > 0) meshes.push(bestMesh)
  }

  if (meshes.length === 0) return null
  if (meshes.length === 1) return meshes[0]

  return combineMeshes(meshes)
}

function extractMeshFromTaggedChunks(buffer: ArrayBuffer): PreviewMeshData | null {
  const bytes = new Uint8Array(buffer)
  const root = parseBigEndianChunks(bytes)
  if (!root) return null

  const flat = flattenChunks(root)
  const vertChunks = flat.filter((chunk) => chunk.tag === 'VERT' && chunk.size >= 12)
  const indexChunks = flat.filter((chunk) => chunk.tag === 'INDX' && chunk.size >= 6)
  if (vertChunks.length === 0 || indexChunks.length === 0) return null

  const meshes: PreviewMeshData[] = []

  for (const v of vertChunks) {
    const vPayload = bytes.subarray(v.dataStart, v.dataEnd)
    const positions = decodeVertexTriples(vPayload)
    const vertexCount = Math.floor(positions.length / 3)
    if (vertexCount < 3) continue

    let bestIndices: number[] | null = null
    let bestScore = Number.NEGATIVE_INFINITY

    for (const i of indexChunks) {
      const iPayload = bytes.subarray(i.dataStart, i.dataEnd)
      const decoded = decodeIndices(iPayload, vertexCount, positions)
      if (decoded.length < 3) continue
      const score = scoreIndexTopology(positions, decoded)
      if (!bestIndices || score > bestScore) {
        bestIndices = decoded
        bestScore = score
      }
    }

    if (!bestIndices) continue

    const pruned = pruneSpikyTriangles({ positions, indices: bestIndices })
    if (!pruned || pruned.indices.length < 3) continue
    meshes.push(pruned)
  }

  return combineMeshes(meshes)
}



interface VertexCandidate {
  positions: number[]
  score: number
}
function buildSequentialIndices(vertexCount: number): number[] {
  const out: number[] = []
  for (let i = 0; i + 2 < vertexCount; i += 3) {
    out.push(i, i + 1, i + 2)
  }
  return out
}

function extractInterleavedVertices(data: Uint8Array, stride: number): VertexCandidate | null {
  if (data.byteLength < stride * 6 || stride < 12) return null
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength)
  const vertexCount = Math.floor(data.byteLength / stride)
  if (vertexCount < 6) return null

  const positions: number[] = []
  let valid = 0
  let minX = Number.POSITIVE_INFINITY
  let minY = Number.POSITIVE_INFINITY
  let minZ = Number.POSITIVE_INFINITY
  let maxX = Number.NEGATIVE_INFINITY
  let maxY = Number.NEGATIVE_INFINITY
  let maxZ = Number.NEGATIVE_INFINITY

  const limit = Math.min(vertexCount, 18000)
  for (let i = 0; i < limit; i += 1) {
    const base = i * stride
    const x = view.getFloat32(base, true)
    const y = view.getFloat32(base + 4, true)
    const z = view.getFloat32(base + 8, true)
    if (!isLikelyVertex(x) || !isLikelyVertex(y) || !isLikelyVertex(z)) continue

    valid += 1
    positions.push(x, y, z)
    if (x < minX) minX = x
    if (y < minY) minY = y
    if (z < minZ) minZ = z
    if (x > maxX) maxX = x
    if (y > maxY) maxY = y
    if (z > maxZ) maxZ = z
  }

  const resultingVertices = Math.floor(positions.length / 3)
  if (resultingVertices < 12) return null

  const spanX = maxX - minX
  const spanY = maxY - minY
  const spanZ = maxZ - minZ
  const span = spanX + spanY + spanZ
  if (!Number.isFinite(span) || span < 0.01) return null

  const density = valid / limit
  if (density < 0.2) return null

  const score = resultingVertices * density + span * 0.01
  return { positions, score }
}

function extractIndexCandidate(data: Uint8Array, vertexCount: number): number[] | null {
  if (vertexCount < 3 || data.byteLength < 6) return null

  const tryU16 = (): number[] | null => {
    const count = Math.floor(data.byteLength / 2)
    if (count < 3) return null
    const view = new DataView(data.buffer, data.byteOffset, data.byteLength)
    const raw: number[] = []
    for (let i = 0; i < count; i += 1) {
      const value = view.getUint16(i * 2, true)
      if (value >= vertexCount) return null
      raw.push(value)
      if (raw.length >= 120000) break
    }

    const out: number[] = []
    for (let i = 0; i + 2 < raw.length; i += 3) {
      const a = raw[i]
      const b = raw[i + 1]
      const c = raw[i + 2]
      if (a === b || b === c || a === c) continue
      out.push(a, b, c)
    }
    return out.length >= 3 ? out : null
  }

  const tryU32 = (): number[] | null => {
    const count = Math.floor(data.byteLength / 4)
    if (count < 3) return null
    const view = new DataView(data.buffer, data.byteOffset, data.byteLength)
    const raw: number[] = []
    for (let i = 0; i < count; i += 1) {
      const value = view.getUint32(i * 4, true)
      if (value >= vertexCount) return null
      raw.push(value)
      if (raw.length >= 120000) break
    }

    const out: number[] = []
    for (let i = 0; i + 2 < raw.length; i += 3) {
      const a = raw[i]
      const b = raw[i + 1]
      const c = raw[i + 2]
      if (a === b || b === c || a === c) continue
      out.push(a, b, c)
    }
    return out.length >= 3 ? out : null
  }

  const u16 = tryU16()
  const u32 = tryU32()
  if (u16 && u32) return u16.length >= u32.length ? u16 : u32
  return u16 ?? u32
}

function buildOutlierVertexMask(positions: number[]): {
  outlier: Uint8Array
  radiusP90Sq: number
} {
  const vertexCount = Math.floor(positions.length / 3)
  const outlier = new Uint8Array(vertexCount)
  if (vertexCount === 0) return { outlier, radiusP90Sq: 0 }

  let minX = Number.POSITIVE_INFINITY
  let minY = Number.POSITIVE_INFINITY
  let minZ = Number.POSITIVE_INFINITY
  let maxX = Number.NEGATIVE_INFINITY
  let maxY = Number.NEGATIVE_INFINITY
  let maxZ = Number.NEGATIVE_INFINITY

  for (let i = 0; i < vertexCount; i += 1) {
    const x = positions[i * 3]
    const y = positions[i * 3 + 1]
    const z = positions[i * 3 + 2]
    if (x < minX) minX = x
    if (y < minY) minY = y
    if (z < minZ) minZ = z
    if (x > maxX) maxX = x
    if (y > maxY) maxY = y
    if (z > maxZ) maxZ = z
  }

  const cx = (minX + maxX) * 0.5
  const cy = (minY + maxY) * 0.5
  const cz = (minZ + maxZ) * 0.5

  const distSq: number[] = new Array(vertexCount)
  for (let i = 0; i < vertexCount; i += 1) {
    const dx = positions[i * 3] - cx
    const dy = positions[i * 3 + 1] - cy
    const dz = positions[i * 3 + 2] - cz
    distSq[i] = dx * dx + dy * dy + dz * dz
  }

  const sorted = [...distSq].sort((a, b) => a - b)
  const p90Index = Math.max(0, Math.min(sorted.length - 1, Math.floor(sorted.length * 0.9)))
  const radiusP90Sq = sorted[p90Index] || 1e-6
  const outlierLimitSq = Math.max(radiusP90Sq * 49, 1e-4)

  for (let i = 0; i < vertexCount; i += 1) {
    if (distSq[i] > outlierLimitSq) outlier[i] = 1
  }

  return { outlier, radiusP90Sq }
}

function triangleArea2(
  ax: number,
  ay: number,
  az: number,
  bx: number,
  by: number,
  bz: number,
  cx: number,
  cy: number,
  cz: number,
): number {
  const abx = bx - ax
  const aby = by - ay
  const abz = bz - az
  const acx = cx - ax
  const acy = cy - ay
  const acz = cz - az
  const nx = aby * acz - abz * acy
  const ny = abz * acx - abx * acz
  const nz = abx * acy - aby * acx
  return nx * nx + ny * ny + nz * nz
}

function edgeLenSq(
  ax: number,
  ay: number,
  az: number,
  bx: number,
  by: number,
  bz: number,
): number {
  const dx = bx - ax
  const dy = by - ay
  const dz = bz - az
  return dx * dx + dy * dy + dz * dz
}

function scoreIndexTopology(positions: number[], indices: number[]): number {
  const vertexCount = Math.floor(positions.length / 3)
  const triCount = Math.floor(indices.length / 3)
  if (vertexCount < 3 || triCount === 0) return Number.NEGATIVE_INFINITY

  const { outlier, radiusP90Sq } = buildOutlierVertexMask(positions)
  const longEdgeLimitSq = Math.max(radiusP90Sq * 64, 1)

  const sampleStep = Math.max(1, Math.floor(triCount / 1800))
  let sampled = 0
  let valid = 0
  let longEdge = 0
  let outlierTri = 0

  for (let ti = 0; ti < triCount; ti += sampleStep) {
    const base = ti * 3
    const ia = indices[base]
    const ib = indices[base + 1]
    const ic = indices[base + 2]
    if (ia < 0 || ib < 0 || ic < 0 || ia >= vertexCount || ib >= vertexCount || ic >= vertexCount) continue
    sampled += 1

    if (outlier[ia] || outlier[ib] || outlier[ic]) {
      outlierTri += 1
      continue
    }

    const ax = positions[ia * 3]
    const ay = positions[ia * 3 + 1]
    const az = positions[ia * 3 + 2]
    const bx = positions[ib * 3]
    const by = positions[ib * 3 + 1]
    const bz = positions[ib * 3 + 2]
    const cx = positions[ic * 3]
    const cy = positions[ic * 3 + 1]
    const cz = positions[ic * 3 + 2]

    const area2 = triangleArea2(ax, ay, az, bx, by, bz, cx, cy, cz)
    if (!Number.isFinite(area2) || area2 < 1e-12) continue
    valid += 1

    const e0 = edgeLenSq(ax, ay, az, bx, by, bz)
    const e1 = edgeLenSq(ax, ay, az, cx, cy, cz)
    const e2 = edgeLenSq(bx, by, bz, cx, cy, cz)
    const maxEdge = Math.max(e0, e1, e2)
    if (maxEdge > longEdgeLimitSq) longEdge += 1
  }

  if (sampled === 0 || valid === 0) return Number.NEGATIVE_INFINITY
  const validRatio = valid / sampled
  const outlierRatio = outlierTri / sampled
  const longEdgeRatio = longEdge / sampled

  return (
    indices.length +
    validRatio * 120000 -
    outlierRatio * 220000 -
    longEdgeRatio * 140000
  )
}

function pruneSpikyTriangles(mesh: PreviewMeshData): PreviewMeshData | null {
  const vertexCount = Math.floor(mesh.positions.length / 3)
  const triCount = Math.floor(mesh.indices.length / 3)
  if (vertexCount < 3 || triCount === 0) return null

  const { outlier, radiusP90Sq } = buildOutlierVertexMask(mesh.positions)
  const longEdgeLimitSq = Math.max(radiusP90Sq * 81, 1)
  const filtered: number[] = []

  for (let ti = 0; ti < triCount; ti += 1) {
    const base = ti * 3
    const ia = mesh.indices[base]
    const ib = mesh.indices[base + 1]
    const ic = mesh.indices[base + 2]
    if (ia < 0 || ib < 0 || ic < 0 || ia >= vertexCount || ib >= vertexCount || ic >= vertexCount) continue
    if (ia === ib || ib === ic || ia === ic) continue
    if (outlier[ia] || outlier[ib] || outlier[ic]) continue

    const ax = mesh.positions[ia * 3]
    const ay = mesh.positions[ia * 3 + 1]
    const az = mesh.positions[ia * 3 + 2]
    const bx = mesh.positions[ib * 3]
    const by = mesh.positions[ib * 3 + 1]
    const bz = mesh.positions[ib * 3 + 2]
    const cx = mesh.positions[ic * 3]
    const cy = mesh.positions[ic * 3 + 1]
    const cz = mesh.positions[ic * 3 + 2]

    const area2 = triangleArea2(ax, ay, az, bx, by, bz, cx, cy, cz)
    if (!Number.isFinite(area2) || area2 < 1e-12) continue

    const e0 = edgeLenSq(ax, ay, az, bx, by, bz)
    const e1 = edgeLenSq(ax, ay, az, cx, cy, cz)
    const e2 = edgeLenSq(bx, by, bz, cx, cy, cz)
    if (Math.max(e0, e1, e2) > longEdgeLimitSq) continue

    filtered.push(ia, ib, ic)
  }

  // Fail closed: do not keep pathological spike geometry.
  if (filtered.length < 3) return null

  return {
    positions: mesh.positions,
    indices: filtered,
    normals: mesh.normals,
    uvs: mesh.uvs,
    uvs1: mesh.uvs1,
    uvSets: mesh.uvSets,
    hasUvChannel: mesh.hasUvChannel,
    decodeDebug: mesh.decodeDebug,
  }
}

function extractMeshFromBytes(
  data: Uint8Array,
  siblings: Uint8Array[],
  _options?: { preferSequential?: boolean },
): PreviewMeshData | null {
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength)
  const floatCount = Math.floor(data.byteLength / 4)
  if (floatCount < 18 && data.byteLength < 72) return null

  const strideCandidates = [12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 56, 64]
  let best: VertexCandidate | null = null

  for (const stride of strideCandidates) {
    if (data.byteLength < stride * 6) continue
    const candidate = extractInterleavedVertices(data, stride)
    if (!candidate) continue
    if (!best || candidate.score > best.score) {
      best = candidate
    }
  }

  if (!best) {
    const positions: number[] = []
    for (let i = 0; i + 2 < floatCount; i += 3) {
      const base = i * 4
      const x = view.getFloat32(base, true)
      const y = view.getFloat32(base + 4, true)
      const z = view.getFloat32(base + 8, true)
      if (!isLikelyVertex(x) || !isLikelyVertex(y) || !isLikelyVertex(z)) continue
      positions.push(x, y, z)
      if (positions.length >= 36000) break
    }
    if (positions.length < 36) return null
    best = { positions, score: positions.length }
  }

  const vertices = Math.floor(best.positions.length / 3)
  if (vertices < 3) return null

  const sequential = buildSequentialIndices(vertices)
  let indices: number[] | null = sequential.length >= 3 ? sequential : null
  let indexScore = indices ? scoreIndexTopology(best.positions, indices) : Number.NEGATIVE_INFINITY

  for (const chunk of siblings) {
    const candidate = extractIndexCandidate(chunk, vertices)
    if (!candidate) continue
    const candidateScore = scoreIndexTopology(best.positions, candidate)
    if (!indices || candidateScore > indexScore || (candidateScore === indexScore && candidate.length > indices.length)) {
      indices = candidate
      indexScore = candidateScore
    }
  }

  if (!indices) {
    indices = sequential
  }

  if (indices.length < 3) return null
  return pruneSpikyTriangles({ positions: best.positions, indices })
}

export function extractMeshFromAsset(buffer: ArrayBuffer, sourcePath?: string): PreviewMeshData | null {
  const sourceExt = sourcePath ? ext(sourcePath) : ''
  const allowPobFirst = sourceExt === '.pob' || sourceExt === ''
  const preferSequentialIndices = false

  if (allowPobFirst) {
    const pob = extractMeshFromPob(buffer)
    if (pob) return pob
  }

  if (sourceExt === '.msh' || sourceExt === '.lod') {
    if (sourceExt === '.msh') {
      const bytes = new Uint8Array(buffer)
      const root = parseBigEndianChunks(bytes)
      if (root) {
        const strictMeshes = extractStructuredMshMeshes(bytes, root)
        if (strictMeshes.length > 0) {
          const merged = combineMeshes(strictMeshes)
          if (merged) return merged
        }
      }
    }

    const tagged = extractMeshFromTaggedChunks(buffer)
    if (tagged) return tagged
  }

  try {
    const root = parseIff(buffer)
    const chunks = walkChunks(root)
    const chunkPayloads = chunks
      .filter((chunk) => Boolean(chunk.data && chunk.data.length >= 64))
      .map((chunk) => chunk.data as Uint8Array)

    let best: PreviewMeshData | null = null

    for (const chunk of chunks) {
      if (!chunk.data || chunk.data.length < 64) continue
      const mesh = extractMeshFromBytes(chunk.data, chunkPayloads, {
        preferSequential: preferSequentialIndices,
      })
      if (!mesh) continue
      if (!best || meshQualityScore(mesh) > meshQualityScore(best)) {
        best = mesh
      }
    }

    if (best) return best
  } catch {
    // Ignore and fallback to full-buffer heuristic below.
  }

  const all = new Uint8Array(buffer)
  const fallback = extractMeshFromBytes(all, [all], {
    preferSequential: preferSequentialIndices,
  })
  if (fallback) return fallback

  // Final fallback for unknown sources: attempt POB extraction once.
  if (sourceExt === '') {
    return extractMeshFromPob(buffer)
  }

  return null
}

async function resolveFirstExistingByExt(
  refs: string[],
  extensions: string[],
  lookup: RepositoryLookup,
): Promise<{ path: string; source: RepositorySourceFile } | null> {
  const candidates = refs
    .filter((value) => extensions.includes(ext(value)))
    .sort((a, b) => {
      const ai = extensions.indexOf(ext(a))
      const bi = extensions.indexOf(ext(b))
      if (ai !== bi) return ai - bi
      return a.localeCompare(b)
    })

  for (const candidate of candidates) {
    const source = await lookup(candidate)
    if (source) return { path: candidate, source }
  }

  return null
}

async function resolveFirstExistingInOrder(
  refs: string[],
  lookup: RepositoryLookup,
): Promise<{ path: string; source: RepositorySourceFile } | null> {
  const seen = new Set<string>()
  const isTexture = refs.length > 0 && (refs[0].endsWith('.dds') || refs[0].endsWith('.tga'))
  
  if (isTexture && refs.length > 0) {
    console.log(`[Candidate Check] Checking ${refs.length} texture candidate(s):`, refs)
  }
  
  for (const raw of refs) {
    const candidate = normalizeSwgPath(raw)
    if (!candidate || seen.has(candidate)) continue
    seen.add(candidate)
    const source = await lookup(candidate)
    if (source) {
      if (isTexture) console.log(`[Candidate Check] \u2713 Selected: ${candidate}`)
      return { path: candidate, source }
    }
  }
  
  if (isTexture) console.log(`[Candidate Check] \u2717 None of the candidates found`)
  return null
}

function parseInlineSwgString(payload: Uint8Array): string {
  const encodings: Array<'u8' | 'u16le' | 'u16be' | 'u32le' | 'u32be' | 'cstr'> = [
    'u8', 'u16le', 'u16be', 'u32le', 'u32be', 'cstr',
  ]
  for (const enc of encodings) {
    const parsed = readInlineStringWithEncoding(payload, 0, enc)
    const value = parsed.value.trim()
    if (!value) continue
    return value
  }
  return ''
}

function readAsciiFourCC(bytes: Uint8Array, offset: number): string | null {
  if (offset < 0 || offset + 4 > bytes.length) return null
  const chars = [bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]]
  if (chars.some((code) => code < 0x20 || code > 0x7e)) return null
  const token = String.fromCharCode(chars[0], chars[1], chars[2], chars[3]).trim()
  return token.length > 0 ? token.toUpperCase() : null
}

function normalizeMaterialToken(raw: string): string {
  const token = raw.trim().toUpperCase()
  if (!token) return ''
  return token
}

function addMaterialHint(token: string | null, target: Set<string>): void {
  if (!token) return
  const normalized = normalizeMaterialToken(token)
  if (!normalized) return
  target.add(normalized)
  // Some IFF dumps display Rtag values reversed; preserve both forms for direct structural matching.
  target.add(normalized.split('').reverse().join(''))
}

function extractMaterialHintsFromScope(
  bytes: Uint8Array,
  scope: BinaryChunk,
): { tags: string[]; names: string[] } {
  const tags = new Set<string>()
  const names = new Set<string>()
  const descendants = flattenChunkSubtree(scope)

  for (const chunk of descendants) {
    if (chunk.tag === 'TAG' && chunk.size >= 4) {
      addMaterialHint(readAsciiFourCC(bytes, chunk.dataStart), tags)
      continue
    }

    if (chunk.tag === 'MATL' && chunk.size >= 2) {
      const payload = bytes.subarray(chunk.dataStart, chunk.dataEnd)
      const parsed = parseInlineSwgString(payload)
      if (parsed) names.add(parsed.trim().toUpperCase())
    }
  }

  return {
    tags: Array.from(tags),
    names: Array.from(names),
  }
}

const shaderMaterialHintsCache = new Map<string, { tags: string[]; names: string[] }>()

function extractShaderMaterialHints(
  shaderBuffer: ArrayBuffer,
  shaderPath: string,
): { tags: string[]; names: string[] } {
  const normalized = normalizeSwgPath(shaderPath)
  const cached = shaderMaterialHintsCache.get(normalized)
  if (cached) return cached

  const bytes = new Uint8Array(shaderBuffer)
  const root = parseBigEndianChunks(bytes)
  if (!root) {
    const empty = { tags: [], names: [] }
    shaderMaterialHintsCache.set(normalized, empty)
    return empty
  }

  const tags = new Set<string>()
  const names = new Set<string>()
  const forms = flattenChunks(root).filter((chunk) => chunk.tag === 'FORM' && chunk.type === 'MATS')
  for (const mats of forms) {
    for (const child of mats.children) {
      if (child.tag !== 'FORM') continue
      const hints = extractMaterialHintsFromScope(bytes, child)
      for (const tag of hints.tags) tags.add(tag)
      for (const name of hints.names) names.add(name)
    }
  }

  const out = {
    tags: Array.from(tags),
    names: Array.from(names),
  }
  shaderMaterialHintsCache.set(normalized, out)
  return out
}

function scoreShaderMaterialMatch(
  partHints: { tags: string[]; names: string[] },
  shaderHints: { tags: string[]; names: string[] },
): number {
  if ((partHints.tags.length === 0 && partHints.names.length === 0) ||
      (shaderHints.tags.length === 0 && shaderHints.names.length === 0)) {
    return 0
  }

  let score = 0
  if (partHints.tags.length > 0 && shaderHints.tags.length > 0) {
    const shaderTags = new Set(shaderHints.tags)
    for (const tag of partHints.tags) {
      if (shaderTags.has(tag)) {
        score += 4
      }
    }
  }

  if (partHints.names.length > 0 && shaderHints.names.length > 0) {
    const shaderNames = new Set(shaderHints.names)
    for (const name of partHints.names) {
      if (shaderNames.has(name)) {
        score += 2
      }
    }
  }

  return score
}

function buildPartShaderTrace(
  shaderPath: string,
  partHints: { tags: string[]; names: string[] },
  shaderHints: { tags: string[]; names: string[] },
  stageRefs: {
    stageTextures: string[]
    stageNormals: string[]
    secondaryTextures: string[]
    primaryUvSetIndex: number
    secondaryUvSetIndex: number
    hasTxms: boolean
    tcssPayloadsParsed: number
    tcssRecognizedAssignments: number
    tcssUniqueAssignments: number
  },
): string {
  const partTagText = partHints.tags.length > 0 ? partHints.tags.join('|') : 'n/a'
  const partNameText = partHints.names.length > 0 ? partHints.names.join('|') : 'n/a'
  const shaderTagText = shaderHints.tags.length > 0 ? shaderHints.tags.join('|') : 'n/a'
  const shaderNameText = shaderHints.names.length > 0 ? shaderHints.names.join('|') : 'n/a'
  const mainText = stageRefs.stageTextures.length > 0 ? stageRefs.stageTextures.join('|') : 'n/a'
  const normalText = stageRefs.stageNormals.length > 0 ? stageRefs.stageNormals.join('|') : 'n/a'
  const detailText = stageRefs.secondaryTextures.length > 0 ? stageRefs.secondaryTextures.join('|') : 'n/a'
  return [
    `shader=${shaderPath}`,
    `txms=${stageRefs.hasTxms ? 'yes' : 'no'}`,
    `partTags=${partTagText}`,
    `partNames=${partNameText}`,
    `shaderTags=${shaderTagText}`,
    `shaderNames=${shaderNameText}`,
    `tcssMain=${stageRefs.primaryUvSetIndex}`,
    `tcssDeta=${stageRefs.secondaryUvSetIndex}`,
    `tcssPayloads=${stageRefs.tcssPayloadsParsed}`,
    `tcssApplied=${stageRefs.tcssRecognizedAssignments}`,
    `tcssUnique=${stageRefs.tcssUniqueAssignments}`,
    `main=${mainText}`,
    `normal=${normalText}`,
    `detail=${detailText}`,
  ].join(' ; ')
}

function isLikelyShaderReferenceName(rawName: string): boolean {
  const normalized = normalizeSwgPath(rawName)
  if (!normalized) return false

  const extension = ext(normalized)
  if (extension === '.sht' || extension === '.sat' || extension === '.trt') return true

  // If there is an explicit extension and it is not a shader family extension,
  // this NAME is not a shader reference.
  if (normalized.includes('.') && extension !== '') return false

  // Skip well-known non-shader chunk references when no extension is present.
  // These often appear in NAME chunks within mesh/shader graphs.
  if (/\.(dds|tga|msh|lod|cmp|apt|pob|eft|vsh|psh)$/i.test(normalized)) return false
  return true
}

function buildShaderCandidatesFromName(basePath: string, rawName: string): string[] {
  const normalizedRaw = normalizeSwgPath(rawName)
  if (!normalizedRaw) return []

  const candidates = normalizedRaw.includes('.')
    ? resolveRefCandidates(basePath, normalizedRaw)
    : [
        ...resolveRefCandidates(basePath, `${normalizedRaw}.sht`),
        normalizeSwgPath(`shader/${normalizedRaw}.sht`),
      ]

  return Array.from(
    new Set(candidates.map((value) => normalizeSwgPath(value)).filter((value) => ext(value) === '.sht')),
  )
}

type ShaderTextureSlot = 'diffuse' | 'normal' | 'other'

function classifyDeclaredTextureSlot(
  texturePath: string,
  prevTag?: string,
  nextTag?: string,
  parentType?: string,
): ShaderTextureSlot {
  // Check IFF structural context first — parent/sibling tag names are declared slot labels
  // that encode the actual role (e.g. MAIN = diffuse, NRML = normal map). These are more
  // authoritative than filename guessing, so they take priority.
  const context = `${prevTag ?? ''} ${nextTag ?? ''} ${parentType ?? ''}`.toUpperCase()
  if (/(LMRN|NRML|NORM|BUMP)/.test(context)) return 'normal'
  if (/(NIAM|MAIN|DIFF|ALBD|COLR|ITXM|KTXM|PTXM)/.test(context)) return 'diffuse'
  // Fall back to filename heuristic only when IFF context is ambiguous.
  if (isLikelyNonDiffuseTexturePath(texturePath)) return 'normal'
  return 'other'
}

function extractShaderDeclaredTextureRefs(
  shaderBuffer: ArrayBuffer,
  shaderPath: string,
): Array<{ path: string; slot: ShaderTextureSlot; order: number }> {
  const bytes = new Uint8Array(shaderBuffer)
  const root = parseBigEndianChunks(bytes)
  if (!root) return []

  const out: Array<{ path: string; slot: ShaderTextureSlot; order: number }> = []
  const seen = new Set<string>()

  const visit = (node: BinaryChunk) => {
    for (let i = 0; i < node.children.length; i += 1) {
      const child = node.children[i]
      if (child.tag === 'NAME' && child.size >= 2) {
        const payload = bytes.subarray(child.dataStart, child.dataEnd)
        const raw = parseInlineSwgString(payload)
        if (raw) {
          const resolved = resolveRefCandidates(shaderPath, raw)
            .map((value) => normalizeSwgPath(value))
            .filter((value) => ['.dds', '.tga'].includes(ext(value)))
          for (const path of resolved) {
            if (seen.has(path)) continue
            seen.add(path)
            out.push({
              path,
              slot: classifyDeclaredTextureSlot(
                path,
                node.children[i - 1]?.tag,
                node.children[i + 1]?.tag,
                node.type,
              ),
              order: out.length,
            })
          }
        }
      }
      visit(child)
    }
  }

  visit(root)
  return out
}

function extractEffectDeclaredTextureRefs(
  effectBuffer: ArrayBuffer,
  effectPath: string,
): Array<{ path: string; slot: ShaderTextureSlot; order: number }> {
  const normalizedPath = normalizeSwgPath(effectPath)
  const cached = effectDeclaredTextureRefsCache.get(normalizedPath)
  if (cached) return cached

  const bytes = new Uint8Array(effectBuffer)
  const root = parseBigEndianChunks(bytes)
  if (!root) return []

  const out: Array<{ path: string; slot: ShaderTextureSlot; order: number }> = []
  const seen = new Set<string>()

  const visit = (node: BinaryChunk) => {
    for (let i = 0; i < node.children.length; i += 1) {
      const child = node.children[i]
      if (child.tag === 'NAME' && child.size >= 2) {
        const payload = bytes.subarray(child.dataStart, child.dataEnd)
        const raw = parseInlineSwgString(payload)
        if (raw) {
          const resolved = resolveRefCandidates(effectPath, raw)
            .map((value) => normalizeSwgPath(value))
            .filter((value) => ['.dds', '.tga'].includes(ext(value)))
          for (const path of resolved) {
            if (seen.has(path)) continue
            seen.add(path)
            out.push({
              path,
              slot: classifyDeclaredTextureSlot(
                path,
                node.children[i - 1]?.tag,
                node.children[i + 1]?.tag,
                node.type,
              ),
              order: out.length,
            })
          }
        }
      }
      visit(child)
    }
  }

  visit(root)
  effectDeclaredTextureRefsCache.set(normalizedPath, out)
  return out
}

function decodeChunkToken(payload: Uint8Array): string | null {
  if (payload.length === 0) return null
  const chars: string[] = []
  for (let i = 0; i < Math.min(payload.length, 32); i += 1) {
    const code = payload[i]
    if (code === 0) break
    if (code < 0x20 || code > 0x7e) continue
    chars.push(String.fromCharCode(code))
  }
  const token = chars.join('').trim()
  return token.length > 0 ? token : null
}

function extractEffectProgramAndOptionRefs(
  effectBuffer: ArrayBuffer,
  effectPath: string,
): { optionCodes: string[]; vertexPrograms: string[]; pixelPrograms: string[] } {
  const normalizedPath = normalizeSwgPath(effectPath)
  const cached = effectRecipeCache.get(normalizedPath)
  if (cached) return cached

  const bytes = new Uint8Array(effectBuffer)
  const root = parseBigEndianChunks(bytes)
  if (!root) return { optionCodes: [], vertexPrograms: [], pixelPrograms: [] }

  const optionCodes: string[] = []
  const vertexPrograms: string[] = []
  const pixelPrograms: string[] = []
  const optionSeen = new Set<string>()
  const vpSeen = new Set<string>()
  const ppSeen = new Set<string>()

  const allChunks = flattenChunks(root)
  for (const chunk of allChunks) {
    if (chunk.tag === 'OPTN' && chunk.size > 0) {
      const payload = bytes.subarray(chunk.dataStart, chunk.dataEnd)
      const token = decodeChunkToken(payload)
      if (token && !optionSeen.has(token)) {
        optionSeen.add(token)
        optionCodes.push(token)
      }
    }

    if (chunk.tag === 'NAME' && chunk.size >= 2) {
      const payload = bytes.subarray(chunk.dataStart, chunk.dataEnd)
      const raw = parseInlineSwgString(payload)
      if (!raw) continue
      const refs = resolveRefCandidates(effectPath, raw)
      for (const ref of refs) {
        const normalized = normalizeSwgPath(ref)
        const e = ext(normalized)
        if (e === '.vsh' && !vpSeen.has(normalized)) {
          vpSeen.add(normalized)
          vertexPrograms.push(normalized)
        }
        if (e === '.psh' && !ppSeen.has(normalized)) {
          ppSeen.add(normalized)
          pixelPrograms.push(normalized)
        }
      }
    }
  }

  const out = { optionCodes, vertexPrograms, pixelPrograms }
  effectRecipeCache.set(normalizedPath, out)
  return out
}

function extractShaderStageTextureRefs(
  shaderBuffer: ArrayBuffer,
  shaderPath: string,
): {
  stageTextures: string[]
  stageNormals: string[]
  hasTxms: boolean
  primaryUvSetIndex: number
  tcssPayloadsParsed: number
  tcssRecognizedAssignments: number
  tcssUniqueAssignments: number
  primaryAddressU?: TextureAddressMode
  primaryAddressV?: TextureAddressMode
  primaryMipmapFilter?: TextureFilterMode
  primaryMinificationFilter?: TextureFilterMode
  primaryMagnificationFilter?: TextureFilterMode
  primaryScaleU?: number
  primaryScaleV?: number
  secondaryTextures: string[]
  secondaryUvSetIndex: number
  secondaryAddressU?: TextureAddressMode
  secondaryAddressV?: TextureAddressMode
  secondaryScaleU?: number
  secondaryScaleV?: number
  normalAddressU?: TextureAddressMode
  normalAddressV?: TextureAddressMode
  normalScaleU?: number
  normalScaleV?: number
  alphaTest: boolean
  alphaReference: number
  transparent: boolean
  alphaBlend: boolean
  effectTags: string[]
  shaderDebugChunks?: string[]  // For debugging
} {
  const normalizedPath = normalizeSwgPath(shaderPath)
  const cached = shaderStageRefsCache.get(normalizedPath)
  if (cached) return cached

  const bytes = new Uint8Array(shaderBuffer)
  const root = parseBigEndianChunks(bytes)
  const empty = {
    stageTextures: [] as string[],
    stageNormals: [] as string[],
    hasTxms: false,
    primaryUvSetIndex: 0,
    tcssPayloadsParsed: 0,
    tcssRecognizedAssignments: 0,
    tcssUniqueAssignments: 0,
    primaryMipmapFilter: undefined,
    primaryMinificationFilter: undefined,
    primaryMagnificationFilter: undefined,
    secondaryTextures: [] as string[],
    secondaryUvSetIndex: 0,
    alphaTest: false,
    alphaReference: 0,
    transparent: false,
    alphaBlend: false,
    effectTags: [] as string[],
  }
  if (!root) return empty

  const decodeAddressMode = (raw: number): TextureAddressMode | undefined => {
    // TXM_DATA template enum (verified from shader editor screenshots):
    // Wrap=0, Mirror=1, Clamp=2.
    if (raw === 0) return 'wrap'
    if (raw === 1) return 'mirror'
    if (raw === 2) return 'clamp'
    // Preserve common extended values when present.
    if (raw === 3) return 'border'
    if (raw === 4) return 'mirroronce'
    return undefined
  }

  const decodeFilterMode = (raw: number): TextureFilterMode | undefined => {
    if (raw === 0) return 'none'
    if (raw === 1) return 'point'
    if (raw === 2) return 'linear'
    if (raw === 3) return 'anisotropic'
    if (raw === 4) return 'flatcubic'
    if (raw === 5) return 'gaussiancubic'
    return undefined
  }

  const normalizeShaderSlotTag = (raw: string): string => {
    const token = raw.toUpperCase()
    const reversed = token.split('').reverse().join('')
    const canonical = (value: string): string => {
      if (value === 'NIAM' || value === 'MAIN') return 'MAIN'
      if (value === 'LMRN' || value === 'NRML' || value === 'NORM') return 'NRML'
      if (value === 'ATED' || value === 'DETA') return 'DETA'
      if (value === 'MVNE' || value === 'ENVM') return 'ENVM'
      if (value === 'CEPS' || value === 'SPEC') return 'SPEC'
      return value
    }

    const direct = canonical(token)
    if (direct !== token) return direct
    return canonical(reversed)
  }

  // SWG SHT slot type tags read from the first 4 bytes of each TXM's DATA chunk.
  // All TXM blocks have the subtype 'TXM ' (with trailing space) — the slot type is NOT in
  // the block name but in the DATA payload. Tags appear in file byte order (not reversed).
  //   NIAM = 'MAIN' reversed → primary diffuse texture    → stageTextures
  //   LMRN = 'NRML' reversed → normal/bump map            → stageNormals
  //   ATED = 'DETA' reversed → detail overlay (MODULATE)  → secondaryTextures
  //   MVNE = 'ENVM' reversed → environment/reflection     → ignored
  //   CEPS = 'SPEC' reversed → specular map               → ignored

  const mainTextures: string[] = []
  const normalTextures: string[] = []
  const detailTextures: string[] = []
  const seenBySlot = {
    MAIN: new Set<string>(),
    NRML: new Set<string>(),
    DETA: new Set<string>(),
  }

  // UV set indices from TCSS (parsed below); defaulting to 0 is safe since
  // NIAM always uses UV set 0 in all observed SWG shaders.
  let primaryUvSetIndex = 0
  let secondaryUvSetIndex = 0
  let normalUvSetIndex = 0
  let primaryAddressU: TextureAddressMode | undefined
  let primaryAddressV: TextureAddressMode | undefined
  let primaryMipmapFilter: TextureFilterMode | undefined
  let primaryMinificationFilter: TextureFilterMode | undefined
  let primaryMagnificationFilter: TextureFilterMode | undefined
  let primaryScaleU: number | undefined
  let primaryScaleV: number | undefined
  let secondaryAddressU: TextureAddressMode | undefined
  let secondaryAddressV: TextureAddressMode | undefined
  let secondaryScaleU: number | undefined
  let secondaryScaleV: number | undefined
  let normalAddressU: TextureAddressMode | undefined
  let normalAddressV: TextureAddressMode | undefined
  let normalScaleU: number | undefined
  let normalScaleV: number | undefined
  let hasTxms = false
  let tcssPayloadsParsed = 0
  let tcssRecognizedAssignments = 0
  const tcssUniqueAssignmentKeys = new Set<string>()
  let txmPrimaryUvSetFallback: number | undefined
  let txmSecondaryUvSetFallback: number | undefined
  let txmNormalUvSetFallback: number | undefined
  
  // Material properties
  let alphaTest = false
  let alphaReference = 0
  let transparent = false
  let alphaBlend = false
  const effectTags: string[] = []
  const shaderDebugChunks: string[] = []  // Capture all chunks for debugging
  
  // Capture chunk structure for debugging.
  // SHT files are small (typically a few KB total). Show full hex for ALL chunks
  // up to 4096 bytes so we never truncate the data needed to diagnose UV/sampler issues.
  function captureChunkStructure(node: BinaryChunk, depth = 0) {
    const indent = '  '.repeat(depth)
    if (node.type) {
      shaderDebugChunks.push(`${indent}${node.tag}:${node.type}`)
    } else {
      const dataSize = node.dataEnd - node.dataStart
      const maxBytesToShow = 4096
      if (dataSize > 0 && dataSize <= maxBytesToShow) {
        const hexBytes: string[] = []
        const asciiBytes: string[] = []
        for (let i = node.dataStart; i < node.dataEnd; i++) {
          const b = bytes[i]
          hexBytes.push(b.toString(16).padStart(2, '0'))
          asciiBytes.push(b >= 0x20 && b <= 0x7e ? String.fromCharCode(b) : '.')
        }
        shaderDebugChunks.push(`${indent}${node.tag} (${dataSize}B) [${hexBytes.join(' ')}] "${asciiBytes.join('')}"`)
      } else {
        shaderDebugChunks.push(`${indent}${node.tag} (${dataSize} bytes - truncated)`)
      }
    }
    for (const child of node.children) {
      captureChunkStructure(child, depth + 1)
    }
  }
  captureChunkStructure(root)
  
  // Track what FORM chunks we encounter
  const formChunks = new Set<string>()

  const applyEffectTag = (raw: string): void => {
    const tagValue = raw.trim().toUpperCase()
    if (!tagValue || !/^[A-Z0-9_]{3,8}$/.test(tagValue)) return
    if (!effectTags.includes(tagValue)) {
      effectTags.push(tagValue)
      console.log(`[Shader Effect] Found TAG: ${tagValue}`)
    }

    if (tagValue === 'ALPH' || tagValue === 'TRNS' || tagValue === 'BLND' || tagValue === 'TRANSPARENT') {
      transparent = true
      alphaBlend = true
    }
    if (tagValue === 'PNCH' || tagValue === 'PUNCHOUT') {
      alphaTest = true
      if (alphaReference === 0) alphaReference = 128
    }
    if (tagValue === 'ADDT' || tagValue === 'ADDITIVE' || tagValue === 'ADD') {
      transparent = true
      alphaBlend = true
    }
  }

  const visit = (node: BinaryChunk) => {
    if (node.tag === 'FORM' && node.type) {
      formChunks.add(node.type)
      // NOTE: ARVS (Alpha Reference Value Source) is NOT a reliable alpha-blend
      // signal on its own. Envmask/spec shaders (e.g. a_envmask_specmap.eft)
      // include ARVS to point at the spec map's alpha channel for env masking,
      // even though the rendered material is opaque. Rely on the effect-name
      // family below for the actual transparency decision.
    }

    // SHT references its effect via a NAME chunk at the top level. Effect
    // file names follow the convention "effect\a_alpha*.eft", "effect\a_add*.eft",
    // "effect\a_trans*.eft" for alpha/additive variants. Detect from the name
    // so we don't have to also parse every .eft to learn the blend mode.
    if (node.tag === 'NAME' && node.size >= 5 && node.size <= 128) {
      let s = ''
      for (let i = node.dataStart; i < node.dataEnd; i++) {
        const b = bytes[i]
        if (b === 0) break
        if (b >= 0x20 && b <= 0x7e) s += String.fromCharCode(b)
      }
      const lower = s.toLowerCase()
      if (lower.endsWith('.eft')) {
        // Common alpha-blended effect families.
        if (
          lower.includes('a_alpha') ||
          lower.includes('a_trans') ||
          lower.includes('a_glass') ||
          lower.includes('a_add')
        ) {
          transparent = true
          alphaBlend = true
        }
      }
    }

    if (node.tag === 'FORM' && node.type === 'TXMS') {
      hasTxms = true
      for (const txm of node.children) {
        if (txm.tag !== 'FORM' || !txm.type?.startsWith('TXM')) continue

        // Structure: FORM:TXM  → FORM:0001 → [DATA (slot tag + flags), NAME (texture path)]
        // NAME chunks are grandchildren of TXM, not direct children.
        for (const versionForm of txm.children) {
          if (versionForm.tag !== 'FORM') continue

          let slotTag = ''
          let texturePath = ''
          let slotAddressU: TextureAddressMode | undefined
          let slotAddressV: TextureAddressMode | undefined
          let slotMipmapFilter: TextureFilterMode | undefined
          let slotMinificationFilter: TextureFilterMode | undefined
          let slotMagnificationFilter: TextureFilterMode | undefined
          let slotUvSetFallback: number | undefined
          for (const child of versionForm.children) {
            if (child.tag === 'DATA' && child.size >= 4 && slotTag === '') {
              // First 4 bytes of DATA identify the slot (NIAM, LMRN, ATED, MVNE, CEPS, …).
              const rawSlotTag = String.fromCharCode(
                bytes[child.dataStart],
                bytes[child.dataStart + 1],
                bytes[child.dataStart + 2],
                bytes[child.dataStart + 3],
              )
              slotTag = normalizeShaderSlotTag(rawSlotTag)
              // TXM DATA layout (verified from template):
              // [0..3] Tag, [4] Placeholder, [5] AddressU, [6] AddressV,
              // [7] AddressW, [8] MipFilter, [9] MinFilter, [10] MagFilter
              // [11+] Unknown - may contain UV mode/generation flags
              
              // Log full DATA payload for analysis
              if (child.size > 11) {
                const dataHex = Array.from(bytes.subarray(child.dataStart, child.dataStart + Math.min(child.size, 32)))
                  .map(b => b.toString(16).padStart(2, '0')).join(' ')
                console.log(`[Shader TXM DATA] ${rawSlotTag} size=${child.size} hex=[${dataHex}]`)
              }
              
              if (child.size >= 7) {
                slotAddressU = decodeAddressMode(bytes[child.dataStart + 5])
                slotAddressV = decodeAddressMode(bytes[child.dataStart + 6])
                if (child.size >= 11) {
                  slotMipmapFilter = decodeFilterMode(bytes[child.dataStart + 8])
                  slotMinificationFilter = decodeFilterMode(bytes[child.dataStart + 9])
                  slotMagnificationFilter = decodeFilterMode(bytes[child.dataStart + 10])
                }
                if (child.size >= 12) {
                  // Fallback UV set source: TXM DATA mode byte (slot-local).
                  // This is used only when TCSS is absent/undecodable.
                  const uvModeRaw = bytes[child.dataStart + 11]
                  const uvSet = uvModeRaw & 0x07
                  if (uvSet <= 7) slotUvSetFallback = uvSet
                }
              } else if (child.size >= 10) {
                // Legacy fallback for older assumptions.
                slotAddressU = decodeAddressMode(bytes[child.dataStart + 8])
                slotAddressV = decodeAddressMode(bytes[child.dataStart + 9])
              }
            }
            if (child.tag === 'NAME' && child.size >= 2 && texturePath === '') {
              const payload = bytes.subarray(child.dataStart, child.dataEnd)
              texturePath = parseInlineSwgString(payload) ?? ''
            }
          }

          if (!texturePath || !slotTag) continue

          const resolved = resolveRefCandidates(shaderPath, texturePath)
            .map((v) => normalizeSwgPath(v))
            .filter((v) => ['.dds', '.tga'].includes(ext(v)))

          for (const ref of resolved) {
            if (slotTag === 'MAIN') {
              if (seenBySlot.MAIN.has(ref)) continue
              seenBySlot.MAIN.add(ref)
              mainTextures.push(ref)        // MAIN → primary diffuse
              if (slotUvSetFallback !== undefined) txmPrimaryUvSetFallback ??= slotUvSetFallback
              primaryAddressU ??= slotAddressU
              primaryAddressV ??= slotAddressV
              primaryMipmapFilter ??= slotMipmapFilter
              primaryMinificationFilter ??= slotMinificationFilter
              primaryMagnificationFilter ??= slotMagnificationFilter
            } else if (slotTag === 'NRML') {
              if (seenBySlot.NRML.has(ref)) continue
              seenBySlot.NRML.add(ref)
              normalTextures.push(ref) // NRML → normal map
              if (slotUvSetFallback !== undefined) txmNormalUvSetFallback ??= slotUvSetFallback
              normalAddressU ??= slotAddressU
              normalAddressV ??= slotAddressV
            } else if (slotTag === 'DETA') {
              if (seenBySlot.DETA.has(ref)) continue
              seenBySlot.DETA.add(ref)
              detailTextures.push(ref)  // DETA → detail overlay blend
              if (slotUvSetFallback !== undefined) txmSecondaryUvSetFallback ??= slotUvSetFallback
              secondaryAddressU ??= slotAddressU
              secondaryAddressV ??= slotAddressV
            }
            // MVNE (ENVM) and CEPS (SPEC) are intentionally ignored
            break
          }
        }
      }
    }

    // TCSS: Texture Coordinate Set assignments — maps each slot to a UV set index.
    // Structure variants observed:
    // 1) FORM:TCSS -> CHUNK:0000 (payload is 5-byte entries [4-byte tag][1-byte uvIdx])
    // 2) FORM:TCSS -> FORM:0000 -> CHUNK:DATA (same payload)
    if (node.tag === 'FORM' && node.type === 'TCSS') {
      const applyTcssEntry = (rawTag: string, uvIndexRaw: number): boolean => {
        const tag = normalizeShaderSlotTag(rawTag)
        const uvIdx = uvIndexRaw & 0x07
        if (tag === 'MAIN') {
          primaryUvSetIndex = uvIdx
          tcssRecognizedAssignments += 1
          tcssUniqueAssignmentKeys.add(`MAIN:${uvIdx}`)
          return true
        }
        if (tag === 'DETA') {
          secondaryUvSetIndex = uvIdx
          tcssRecognizedAssignments += 1
          tcssUniqueAssignmentKeys.add(`DETA:${uvIdx}`)
          return true
        }
        if (tag === 'NRML') {
          normalUvSetIndex = uvIdx
          tcssRecognizedAssignments += 1
          tcssUniqueAssignmentKeys.add(`NRML:${uvIdx}`)
          return true
        }
        return false
      }

      const parseTcssPayload = (payload: Uint8Array): boolean => {
        let applied = false
        const view = new DataView(payload.buffer, payload.byteOffset, payload.byteLength)

        const parse5ByteEntries = (startOffset: number) => {
          const size = payload.length - startOffset
          if (size < 5 || size % 5 !== 0) return
          const entryCount = Math.floor(size / 5)
          console.log(`[TCSS 5-byte] Parsing ${entryCount} 5-byte entries at offset ${startOffset} in shader ${shaderPath}`)
          for (let i = 0; i < entryCount; i += 1) {
            const off = startOffset + i * 5
            const rawTag = String.fromCharCode(payload[off], payload[off + 1], payload[off + 2], payload[off + 3])
            const byte4 = payload[off + 4]
            console.log(`[TCSS 5-byte] Entry ${i}: tag="${rawTag}" byte4=${byte4} (normalized: ${normalizeShaderSlotTag(rawTag)})`)
            if (applyTcssEntry(rawTag, payload[off + 4])) {
              applied = true
            }
          }
        }

        const parse8ByteEntries = (startOffset: number) => {
          const size = payload.length - startOffset
          if (size < 8 || size % 8 !== 0) return
          const entryCount = Math.floor(size / 8)
          console.log(`[TCSS 8-byte] Parsing ${entryCount} 8-byte entries at offset ${startOffset}`)
          for (let i = 0; i < entryCount; i += 1) {
            const off = startOffset + i * 8
            const rawTag = String.fromCharCode(payload[off], payload[off + 1], payload[off + 2], payload[off + 3])
            // Observed variants use either byte or uint32 for index.
            const uvLe = view.getUint32(off + 4, true)
            const uvBe = view.getUint32(off + 4, false)
            const byte4 = payload[off + 4]
            const byte5 = payload[off + 5]
            const byte6 = payload[off + 6]
            const byte7 = payload[off + 7]
            console.log(`[TCSS 8-byte] Entry ${i}: tag="${rawTag}" bytes[4-7]=[${byte4}, ${byte5}, ${byte6}, ${byte7}] uvLe=${uvLe} uvBe=${uvBe}`)
            if (applyTcssEntry(rawTag, uvLe)) {
              applied = true
              continue
            }
            if (applyTcssEntry(rawTag, uvBe)) {
              applied = true
            }
          }
        }

        // Try direct payload and common header-prefixed variants.
        parse5ByteEntries(0)
        parse5ByteEntries(4)
        parse8ByteEntries(0)
        parse8ByteEntries(4)

        // Robust fallback: only use sliding token scan when structured layouts yielded nothing.
        if (!applied) {
          for (let off = 0; off + 4 < payload.length; off += 1) {
            const rawTag = String.fromCharCode(payload[off], payload[off + 1], payload[off + 2], payload[off + 3])
            const normalized = normalizeShaderSlotTag(rawTag)
            if (normalized !== 'MAIN' && normalized !== 'DETA' && normalized !== 'NRML') continue
            const uvIndexRaw = payload[off + 4]
            if (applyTcssEntry(rawTag, uvIndexRaw)) applied = true
          }
        }

        return applied
      }

      // Walk the TCSS subtree and parse all non-FORM payload chunks.
      const stack = [...node.children]
      while (stack.length > 0) {
        const next = stack.pop()
        if (!next) continue
        if (next.tag === 'FORM') {
          for (const child of next.children) stack.push(child)
          continue
        }
        if (next.dataEnd <= next.dataStart || next.size < 5) continue
        tcssPayloadsParsed += 1
        parseTcssPayload(bytes.subarray(next.dataStart, next.dataEnd))
      }
    }

    // Parse TCSC (Texture Coordinate Scale) - UV scale multipliers per texture stage
    if (node.tag === 'FORM' && node.type === 'TCSC') {
      console.log(`[TCSC] Found TCSC chunk in shader: ${shaderPath}`)
      const parseTcscPayload = (payload: Uint8Array): void => {
        const view = new DataView(payload.buffer, payload.byteOffset, payload.byteLength)
        console.log(`[TCSC] Parsing payload, length: ${payload.length}`)
        
        // TCSC format: 4-byte tag + 4-byte float scaleU + 4-byte float scaleV (12 bytes per entry)
        const parseEntries = (startOffset: number) => {
          const size = payload.length - startOffset
          if (size < 12 || size % 12 !== 0) {
            console.log(`[TCSC] Size check failed at offset ${startOffset}: size=${size}, not divisible by 12`)
            return
          }
          const entryCount = Math.floor(size / 12)
          console.log(`[TCSC] Parsing ${entryCount} entries at offset ${startOffset}`)
          for (let i = 0; i < entryCount; i += 1) {
            const off = startOffset + i * 12
            const rawTag = String.fromCharCode(payload[off], payload[off + 1], payload[off + 2], payload[off + 3])
            const slotTag = normalizeShaderSlotTag(rawTag)
            const scaleU = view.getFloat32(off + 4, false)  // Big-endian (SWG format)
            const scaleV = view.getFloat32(off + 8, false)  // Big-endian (SWG format)
            
            console.log(`[TCSC] Entry ${i}: rawTag="${rawTag}" slotTag="${slotTag}" scaleU=${scaleU} scaleV=${scaleV}`)
            
            if (Number.isFinite(scaleU) && Number.isFinite(scaleV) && scaleU > 0 && scaleV > 0) {
              if (slotTag === 'MAIN') {
                primaryScaleU ??= scaleU
                primaryScaleV ??= scaleV
                console.log(`[TCSC] ✓ Assigned MAIN scales: U=${scaleU}, V=${scaleV}`)
              } else if (slotTag === 'NRML') {
                normalScaleU ??= scaleU
                normalScaleV ??= scaleV
                console.log(`[TCSC] ✓ Assigned NRML scales: U=${scaleU}, V=${scaleV}`)
              } else if (slotTag === 'DETA') {
                secondaryScaleU ??= scaleU
                secondaryScaleV ??= scaleV
                console.log(`[TCSC] ✓ Assigned DETA scales: U=${scaleU}, V=${scaleV}`)
              }
            } else {
              console.log(`[TCSC] × Invalid scale values, skipping`)
            }
          }
        }

        // Try direct payload and common header-prefixed variants
        parseEntries(0)
        parseEntries(4)
      }

      // Walk the TCSC subtree and parse all non-FORM payload chunks
      const stack = [...node.children]
      while (stack.length > 0) {
        const next = stack.pop()
        if (!next) continue
        if (next.tag === 'FORM') {
          for (const child of next.children) stack.push(child)
          continue
        }
        if (next.dataEnd <= next.dataStart || next.size < 12) continue
        parseTcscPayload(bytes.subarray(next.dataStart, next.dataEnd))
      }
    }

    // Parse ALPH (Alpha Test Reference) chunk
    if (node.tag === 'DATA' && node.size >= 4) {
      const parent = findParentChunk(root, node)
      if (parent?.tag === 'FORM' && parent?.type === 'ALPH') {
        // ALPH DATA contains alpha reference value (0-255)
        const alphaRefValue = bytes[node.dataStart]
        if (alphaRefValue > 0) {
          alphaTest = true
          alphaReference = alphaRefValue
          console.log(`[Shader Alpha] Found ALPH chunk: reference=${alphaRefValue}`)
        }
      }
    }

    // Parse EFCT/TAG effect tags
    if (node.tag === 'TAG ' && node.size === 4) {
      const tagRaw = String.fromCharCode(
        bytes[node.dataStart],
        bytes[node.dataStart + 1],
        bytes[node.dataStart + 2],
        bytes[node.dataStart + 3]
      )
      applyEffectTag(tagRaw)
    }

    if (node.tag === 'DATA' && node.size >= 4) {
      const parent = findParentChunk(root, node)
      if (parent?.tag === 'FORM' && parent.type === 'EFCT') {
        const payload = bytes.subarray(node.dataStart, node.dataEnd)
        // EFCT payloads can encode 4-byte tag tokens and/or inline ASCII tag names.
        for (let i = 0; i + 3 < payload.length; i += 4) {
          const token = String.fromCharCode(payload[i], payload[i + 1], payload[i + 2], payload[i + 3])
          applyEffectTag(token)
        }

        const ascii = extractAsciiStrings(
          payload.buffer.slice(payload.byteOffset, payload.byteOffset + payload.byteLength),
          3,
        )
        for (const str of ascii) {
          applyEffectTag(str)
        }
      }
    }

    for (const child of node.children) visit(child)
  }

  // Helper to find parent chunk (simple implementation)
  function findParentChunk(root: BinaryChunk, target: BinaryChunk): BinaryChunk | null {
    function search(node: BinaryChunk): BinaryChunk | null {
      for (const child of node.children) {
        if (child === target) return node
        const found = search(child)
        if (found) return found
      }
      return null
    }
    return search(root)
  }

  visit(root)

  // Fallback UV set assignment from TXM DATA when TCSS is absent or undecodable.
  if (tcssRecognizedAssignments === 0) {
    if (txmPrimaryUvSetFallback !== undefined) primaryUvSetIndex = txmPrimaryUvSetFallback
    if (txmSecondaryUvSetFallback !== undefined) secondaryUvSetIndex = txmSecondaryUvSetFallback
    if (txmNormalUvSetFallback !== undefined) normalUvSetIndex = txmNormalUvSetFallback
  }

  // normalUvSetIndex is recorded for future use but not yet wired into the result
  void normalUvSetIndex
  
  console.log(`[Shader Chunks] ${shaderPath} contains FORM chunks: ${Array.from(formChunks).join(', ')}`)
  if (primaryScaleU || primaryScaleV || secondaryScaleU || secondaryScaleV || normalScaleU || normalScaleV) {
    console.log(`[Shader Scales] Found scales - primary: [${primaryScaleU ?? 'none'}, ${primaryScaleV ?? 'none'}], secondary: [${secondaryScaleU ?? 'none'}, ${secondaryScaleV ?? 'none'}], normal: [${normalScaleU ?? 'none'}, ${normalScaleV ?? 'none'}]`)
  }

  const out = {
    stageTextures: mainTextures,
    stageNormals: normalTextures,
    hasTxms,
    primaryUvSetIndex,
    tcssPayloadsParsed,
    tcssRecognizedAssignments,
    tcssUniqueAssignments: tcssUniqueAssignmentKeys.size,
    primaryAddressU,
    primaryAddressV,
    primaryMipmapFilter,
    primaryMinificationFilter,
    primaryMagnificationFilter,
    primaryScaleU,
    primaryScaleV,
    secondaryTextures: detailTextures,
    secondaryUvSetIndex,
    secondaryAddressU,
    secondaryAddressV,
    secondaryScaleU,
    secondaryScaleV,
    normalAddressU,
    normalAddressV,
    normalScaleU,
    normalScaleV,
    alphaTest,
    alphaReference,
    transparent,
    alphaBlend,
    effectTags,
    shaderDebugChunks,
  }
  shaderStageRefsCache.set(normalizedPath, out)
  return out
}

function extractMshShaderRefs(buffer: ArrayBuffer, basePath: string): string[] {
  const bytes = new Uint8Array(buffer)
  const root = parseBigEndianChunks(bytes)
  if (!root) return []

  const refs = new Set<string>()
  const names = flattenChunks(root).filter((chunk) => chunk.tag === 'NAME' && chunk.size >= 2)

  for (const nameChunk of names) {
    const payload = bytes.subarray(nameChunk.dataStart, nameChunk.dataEnd)
    const parsed = parseInlineSwgString(payload)
    if (!parsed) continue
    if (!isLikelyShaderReferenceName(parsed)) continue

    const candidates = buildShaderCandidatesFromName(basePath, parsed)
    for (const next of candidates) {
      refs.add(next)
    }
  }

  return Array.from(refs)
}

async function resolveMshPartMeshesWithTextures(
  meshPath: string,
  lookup: RepositoryLookup,
  options: ResolveTemplateOptions = {},
): Promise<{
  parts: PreviewMeshData[]
  texturePaths: string[]
  shaderPaths: (string | undefined)[]
  textureAddressU: (TextureAddressMode | undefined)[]
  textureAddressV: (TextureAddressMode | undefined)[]
  textureMipmapFilter: (TextureFilterMode | undefined)[]
  textureMinificationFilter: (TextureFilterMode | undefined)[]
  textureMagnificationFilter: (TextureFilterMode | undefined)[]
  normalTexturePaths: (string | undefined)[]
  normalTextureAddressU: (TextureAddressMode | undefined)[]
  normalTextureAddressV: (TextureAddressMode | undefined)[]
  primaryUvSetIndices: number[]
  primaryScaleU: (number | undefined)[]
  primaryScaleV: (number | undefined)[]
  secondaryTexturePaths: (string | undefined)[]
  secondaryTextureAddressU: (TextureAddressMode | undefined)[]
  secondaryTextureAddressV: (TextureAddressMode | undefined)[]
  secondaryScaleU: (number | undefined)[]
  secondaryScaleV: (number | undefined)[]
  secondaryUvSetIndices: number[]
  chunkTrace: (string | undefined)[]
  effectPaths: (string | undefined)[]
} | null> {
  const source = await lookup(meshPath)
  if (!source) return null
  const strictDeclaredOnly = options.strictDeclaredOnly === true

  const buffer = await source.file.arrayBuffer()
  const bytes = new Uint8Array(buffer)
  const root = parseBigEndianChunks(bytes)
  if (!root) return null

  const forms = flattenChunks(root).filter((chunk) => chunk.tag === 'FORM')
  const spsRoot = forms.find((chunk) => chunk.type === 'SPS ')
  if (!spsRoot) return null

  const spsVersion = spsRoot.children.find((child) => child.tag === 'FORM')
  if (!spsVersion) return null

  const partForms = spsVersion.children.filter(
    (child) => child.tag === 'FORM' && !!child.type && /^\d{4}$/.test(child.type),
  )
  if (partForms.length === 0) return null

  const parts: PreviewMeshData[] = []
  const texturePaths: string[] = []
  const shaderPaths: (string | undefined)[] = []
  const textureAddressU: (TextureAddressMode | undefined)[] = []
  const textureAddressV: (TextureAddressMode | undefined)[] = []
  const textureMipmapFilter: (TextureFilterMode | undefined)[] = []
  const textureMinificationFilter: (TextureFilterMode | undefined)[] = []
  const textureMagnificationFilter: (TextureFilterMode | undefined)[] = []
  const normalTexturePaths: (string | undefined)[] = []
  const normalTextureAddressU: (TextureAddressMode | undefined)[] = []
  const normalTextureAddressV: (TextureAddressMode | undefined)[] = []
  const primaryUvSetIndices: number[] = []
  const primaryScaleU: (number | undefined)[] = []
  const primaryScaleV: (number | undefined)[] = []
  const secondaryTexturePaths: (string | undefined)[] = []
  const secondaryTextureAddressU: (TextureAddressMode | undefined)[] = []
  const secondaryTextureAddressV: (TextureAddressMode | undefined)[] = []
  const secondaryScaleU: (number | undefined)[] = []
  const secondaryScaleV: (number | undefined)[] = []
  const secondaryUvSetIndices: number[] = []
  const chunkTrace: (string | undefined)[] = []
  const effectPaths: (string | undefined)[] = []

  for (const partForm of partForms) {
    const mesh = decodeStructuredMeshFromScope(bytes, partForm)
    if (!mesh || mesh.positions.length < 9 || mesh.indices.length < 3) continue
    const normalizedMesh = normalizeCandidateMesh(mesh)
    if (!normalizedMesh) continue

    let partTexturePath = ''
    let partTextureAddressU: TextureAddressMode | undefined = undefined
    let partTextureAddressV: TextureAddressMode | undefined = undefined
    let partTextureMipmapFilter: TextureFilterMode | undefined = undefined
    let partTextureMinificationFilter: TextureFilterMode | undefined = undefined
    let partTextureMagnificationFilter: TextureFilterMode | undefined = undefined
    let partPrimaryScaleU: number | undefined = undefined
    let partPrimaryScaleV: number | undefined = undefined
    let partNormalTexturePath: string | undefined = undefined
    let partNormalTextureAddressU: TextureAddressMode | undefined = undefined
    let partNormalTextureAddressV: TextureAddressMode | undefined = undefined
    let partPrimaryUvSetIndex = 0
    let partSecondaryTexturePath: string | undefined = undefined
    let partSecondaryTextureAddressU: TextureAddressMode | undefined = undefined
    let partSecondaryTextureAddressV: TextureAddressMode | undefined = undefined
    let partSecondaryScaleU: number | undefined = undefined
    let partSecondaryScaleV: number | undefined = undefined
    let partSecondaryUvSetIndex = 0
    let partChunkTrace: string | undefined = undefined
    let partShaderPath: string | undefined = undefined
    const partDesc = flattenChunkSubtree(partForm)
    const partMaterialHints = extractMaterialHintsFromScope(bytes, partForm)
    const nameChunks = partDesc.filter((chunk) => chunk.tag === 'NAME' && chunk.size >= 2)
    if (nameChunks.length > 0) {
      const partShaderCandidates: string[] = []
      const seenPartShaders = new Set<string>()
      for (const nameChunk of nameChunks) {
        const payload = bytes.subarray(nameChunk.dataStart, nameChunk.dataEnd)
        const shaderName = parseInlineSwgString(payload)
        if (!shaderName) continue
        if (!isLikelyShaderReferenceName(shaderName)) continue

        const shaderCandidates = buildShaderCandidatesFromName(meshPath, shaderName)
        for (const shaderCandidate of shaderCandidates) {
          if (seenPartShaders.has(shaderCandidate)) continue
          seenPartShaders.add(shaderCandidate)
          partShaderCandidates.push(shaderCandidate)
        }
      }

      const rankedShaderCandidates: Array<{
        path: string
        score: number
        order: number
        shaderHints: { tags: string[]; names: string[] }
        stageRefs: ReturnType<typeof extractShaderStageTextureRefs>
      }> = []
      for (const shaderCandidate of partShaderCandidates) {
        const shaderSource = await lookup(shaderCandidate)
        if (!shaderSource) continue
        const shaderBuffer = await shaderSource.file.arrayBuffer()
        const shaderHints = extractShaderMaterialHints(shaderBuffer, shaderCandidate)
        const stageRefs = extractShaderStageTextureRefs(shaderBuffer, shaderCandidate)
        rankedShaderCandidates.push({
          path: shaderCandidate,
          score: scoreShaderMaterialMatch(partMaterialHints, shaderHints),
          order: partShaderCandidates.indexOf(shaderCandidate),
          shaderHints,
          stageRefs,
        })
      }

      if (!strictDeclaredOnly) {
        rankedShaderCandidates.sort((a, b) => {
          if (a.score !== b.score) return b.score - a.score
          return a.order - b.order
        })
      } else {
        rankedShaderCandidates.sort((a, b) => a.order - b.order)
      }

      for (const ranked of rankedShaderCandidates) {
        const textureHit = await resolveTextureFromShaderPath(ranked.path, lookup, options)
        if (!textureHit) continue
        partTexturePath = textureHit.texturePath
        partTextureAddressU = textureHit.textureAddressU
        partTextureAddressV = textureHit.textureAddressV
        partTextureMipmapFilter = textureHit.textureMipmapFilter
        partTextureMinificationFilter = textureHit.textureMinificationFilter
        partTextureMagnificationFilter = textureHit.textureMagnificationFilter
        partPrimaryScaleU = textureHit.primaryScaleU
        partPrimaryScaleV = textureHit.primaryScaleV
        partNormalTexturePath = textureHit.normalTexturePath
        partNormalTextureAddressU = textureHit.normalTextureAddressU
        partNormalTextureAddressV = textureHit.normalTextureAddressV
        partPrimaryUvSetIndex = textureHit.primaryUvSetIndex ?? 0
        partSecondaryTexturePath = textureHit.secondaryTexturePath
        partSecondaryTextureAddressU = textureHit.secondaryTextureAddressU
        partSecondaryTextureAddressV = textureHit.secondaryTextureAddressV
        partSecondaryScaleU = textureHit.secondaryScaleU
        partSecondaryScaleV = textureHit.secondaryScaleV
        partSecondaryUvSetIndex = textureHit.secondaryUvSetIndex ?? 0
        partChunkTrace = buildPartShaderTrace(
          ranked.path,
          partMaterialHints,
          ranked.shaderHints,
          ranked.stageRefs,
        )
        partShaderPath = ranked.path
        break
      }

      if (!partTexturePath) {
        for (const shaderCandidate of partShaderCandidates) {
          const textureHit = await resolveTextureFromShaderPath(shaderCandidate, lookup, options)
          if (!textureHit) continue
          partTexturePath = textureHit.texturePath
          partTextureAddressU = textureHit.textureAddressU
          partTextureAddressV = textureHit.textureAddressV
          partTextureMipmapFilter = textureHit.textureMipmapFilter
          partTextureMinificationFilter = textureHit.textureMinificationFilter
          partTextureMagnificationFilter = textureHit.textureMagnificationFilter
          partPrimaryScaleU = textureHit.primaryScaleU
          partPrimaryScaleV = textureHit.primaryScaleV
          partNormalTexturePath = textureHit.normalTexturePath
          partNormalTextureAddressU = textureHit.normalTextureAddressU
          partNormalTextureAddressV = textureHit.normalTextureAddressV
          partPrimaryUvSetIndex = textureHit.primaryUvSetIndex ?? 0
          partSecondaryTexturePath = textureHit.secondaryTexturePath
          partSecondaryTextureAddressU = textureHit.secondaryTextureAddressU
          partSecondaryTextureAddressV = textureHit.secondaryTextureAddressV
          partSecondaryScaleU = textureHit.secondaryScaleU
          partSecondaryScaleV = textureHit.secondaryScaleV
          partSecondaryUvSetIndex = textureHit.secondaryUvSetIndex ?? 0
          const shaderSource = await lookup(shaderCandidate)
          if (shaderSource) {
            const shaderBuffer = await shaderSource.file.arrayBuffer()
            const shaderHints = extractShaderMaterialHints(shaderBuffer, shaderCandidate)
            const stageRefs = extractShaderStageTextureRefs(shaderBuffer, shaderCandidate)
            partChunkTrace = buildPartShaderTrace(
              shaderCandidate,
              partMaterialHints,
              shaderHints,
              stageRefs,
            )
          }
          partShaderPath = shaderCandidate
          break
        }
      }
    }

    parts.push(normalizedMesh)
    texturePaths.push(partTexturePath)
    shaderPaths.push(partShaderPath)
    textureAddressU.push(partTextureAddressU)
    textureAddressV.push(partTextureAddressV)
    textureMipmapFilter.push(partTextureMipmapFilter)
    textureMinificationFilter.push(partTextureMinificationFilter)
    textureMagnificationFilter.push(partTextureMagnificationFilter)
    normalTexturePaths.push(partNormalTexturePath)
    normalTextureAddressU.push(partNormalTextureAddressU)
    normalTextureAddressV.push(partNormalTextureAddressV)
    primaryUvSetIndices.push(partPrimaryUvSetIndex)
    primaryScaleU.push(partPrimaryScaleU)
    primaryScaleV.push(partPrimaryScaleV)
    secondaryTexturePaths.push(partSecondaryTexturePath)
    secondaryTextureAddressU.push(partSecondaryTextureAddressU)
    secondaryTextureAddressV.push(partSecondaryTextureAddressV)
    secondaryScaleU.push(partSecondaryScaleU)
    secondaryScaleV.push(partSecondaryScaleV)
    secondaryUvSetIndices.push(partSecondaryUvSetIndex)
    chunkTrace.push(partChunkTrace)

    // Resolve & cache the effect (.eft) file referenced by the part's shader so the
    // surface-pick dump can show the effect chunks (which define UV transforms / DOT3 / blend ops).
    let partEffectPath: string | undefined
    if (partShaderPath) {
      try {
        const shaderSource = await lookup(partShaderPath)
        if (shaderSource) {
          const shaderBuffer = await shaderSource.file.arrayBuffer()
          const shaderRefs = getResolvedRefCandidatesCached(partShaderPath, shaderBuffer)
          const effectHit = await resolveFirstExistingByExt(shaderRefs, ['.eft'], lookup)
          if (effectHit) {
            partEffectPath = effectHit.path
            const effectBuffer = await effectHit.source.file.arrayBuffer()
            captureEffectDebugChunks(effectBuffer, effectHit.path)
          }
        }
      } catch {
        // Non-fatal: effect resolution is debug-only.
      }
    }
    effectPaths.push(partEffectPath)
  }

  if (parts.length === 0) return null
  return {
    parts,
    texturePaths,
    shaderPaths,
    textureAddressU,
    textureAddressV,
    textureMipmapFilter,
    textureMinificationFilter,
    textureMagnificationFilter,
    normalTexturePaths,
    normalTextureAddressU,
    normalTextureAddressV,
    primaryUvSetIndices,
    primaryScaleU,
    primaryScaleV,
    secondaryTexturePaths,
    secondaryTextureAddressU,
    secondaryTextureAddressV,
    secondaryScaleU,
    secondaryScaleV,
    secondaryUvSetIndices,
    chunkTrace,
    effectPaths,
  }
}

async function resolveTextureFromShaderPath(
  shaderPath: string,
  lookup: RepositoryLookup,
  _options: ResolveTemplateOptions = {},
): Promise<{
  texturePath: string
  textureSourceLabel: string
  textureAddressU?: TextureAddressMode
  textureAddressV?: TextureAddressMode
  textureMipmapFilter?: TextureFilterMode
  textureMinificationFilter?: TextureFilterMode
  textureMagnificationFilter?: TextureFilterMode
  primaryScaleU?: number
  primaryScaleV?: number
  normalTexturePath?: string
  normalTextureSourceLabel?: string
  normalTextureAddressU?: TextureAddressMode
  normalTextureAddressV?: TextureAddressMode
  primaryUvSetIndex: number
  secondaryTexturePath?: string
  secondaryUvSetIndex: number
  secondaryTextureAddressU?: TextureAddressMode
  secondaryTextureAddressV?: TextureAddressMode
  secondaryScaleU?: number
  secondaryScaleV?: number
} | null> {
  const cacheKey = normalizeSwgPath(shaderPath)
  const cached = shaderTextureResolutionCache.get(cacheKey)
  if (cached) return cached

  const pending = (async (): Promise<ShaderResolutionResult> => {
    const shaderSource = await lookup(shaderPath)
    if (!shaderSource) return null

    const shaderBuffer = await shaderSource.file.arrayBuffer()
    const shaderRefs = getResolvedRefCandidatesCached(shaderPath, shaderBuffer)

    const stageRefs = extractShaderStageTextureRefs(shaderBuffer, shaderPath)
    console.log(`[Texture Debug] Shader: ${shaderPath}`)
    console.log(`[Texture Debug] Stage textures (MAIN):`, stageRefs.stageTextures)
    console.log(`[Texture Debug] Stage normals (NRML):`, stageRefs.stageNormals)
    console.log(`[Texture Debug] Secondary textures (ATED):`, stageRefs.secondaryTextures)
    // stageTextures/stageNormals are slot-type–identified (NIAM/LMRN from DATA bytes).
    // No filename heuristic filtering needed — slot type IS the classification.
    const stageNormalHit = await resolveFirstExistingInOrder(stageRefs.stageNormals, lookup)
    const stageDiffuseHit = await resolveFirstExistingInOrder(stageRefs.stageTextures, lookup)
    if (stageDiffuseHit) {
      console.log(`[Texture Debug] ✓ Resolved MAIN texture: ${stageDiffuseHit.path}`)
      if (stageNormalHit) console.log(`[Texture Debug] ✓ Resolved NRML texture: ${stageNormalHit.path}`)
      // Resolve secondary (ATED/detail) blend texture when present.
      // No theme-mismatch filter here — ATED detail textures like impl_floor_a_dirt_4way.dds
      // are explicitly identified by slot type and are always intentional.
      const secondaryHit = stageRefs.secondaryTextures.length > 0
        ? await resolveFirstExistingInOrder(stageRefs.secondaryTextures, lookup)
        : null
      if (secondaryHit) console.log(`[Texture Debug] ✓ Resolved ATED texture: ${secondaryHit.path}`)
      return {
        texturePath: stageDiffuseHit.path,
        textureSourceLabel: stageDiffuseHit.source.sourceLabel,
        textureAddressU: stageRefs.primaryAddressU,
        textureAddressV: stageRefs.primaryAddressV,
        textureMipmapFilter: stageRefs.primaryMipmapFilter,
        textureMinificationFilter: stageRefs.primaryMinificationFilter,
        textureMagnificationFilter: stageRefs.primaryMagnificationFilter,
        primaryScaleU: stageRefs.primaryScaleU,
        primaryScaleV: stageRefs.primaryScaleV,
        normalTexturePath: stageNormalHit?.path,
        normalTextureSourceLabel: stageNormalHit?.source.sourceLabel,
        normalTextureAddressU: stageRefs.normalAddressU,
        normalTextureAddressV: stageRefs.normalAddressV,
        primaryUvSetIndex: stageRefs.primaryUvSetIndex,
        secondaryTexturePath: secondaryHit?.path,
        secondaryUvSetIndex: stageRefs.secondaryUvSetIndex,
        secondaryTextureAddressU: stageRefs.secondaryAddressU,
        secondaryTextureAddressV: stageRefs.secondaryAddressV,
        secondaryScaleU: stageRefs.secondaryScaleU,
        secondaryScaleV: stageRefs.secondaryScaleV,
      }
    }

    // If TXMS exists in the shader, treat stage parsing as authoritative.
    // Do not fall through to declared/effect heuristics, which can pick unrelated textures.
    if (stageRefs.hasTxms) {
      console.log(`[Texture Debug] Shader has TXMS but no valid MAIN texture found`)
      return null
    }

    console.log(`[Texture Debug] No TXMS found, checking declared texture refs...`)
    const declaredTextureRefs = extractShaderDeclaredTextureRefs(shaderBuffer, shaderPath)
    const declaredDiffuse = declaredTextureRefs
      .filter((value) => value.slot === 'diffuse')
      .map((value) => value.path)
    const declaredNormal = declaredTextureRefs
      .filter((value) => value.slot === 'normal')
      .map((value) => value.path)

    const declaredNormalHit = await resolveFirstExistingInOrder(declaredNormal, lookup)

    const declaredDiffuseHit = await resolveFirstExistingInOrder(declaredDiffuse, lookup)
    if (declaredDiffuseHit) {
      console.log(`[Texture Debug] ✓ Resolved declared diffuse: ${declaredDiffuseHit.path}`)
      if (declaredNormalHit) console.log(`[Texture Debug] ✓ Resolved declared normal: ${declaredNormalHit.path}`)
      return {
        texturePath: declaredDiffuseHit.path,
        textureSourceLabel: declaredDiffuseHit.source.sourceLabel,
        normalTexturePath: declaredNormalHit?.path,
        normalTextureSourceLabel: declaredNormalHit?.source.sourceLabel,
        primaryUvSetIndex: 0,
        secondaryUvSetIndex: 0,
      }
    }

    console.log(`[Texture Debug] No declared textures, checking effect file...`)
    const effectHit = await resolveFirstExistingByExt(shaderRefs, ['.eft'], lookup)
    if (!effectHit) {
      console.log(`[Texture Debug] No effect file found`)
      return null
    }
    console.log(`[Texture Debug] Found effect: ${effectHit.path}`)

    const effectBuffer = await effectHit.source.file.arrayBuffer()
    const declaredEffectRefs = extractEffectDeclaredTextureRefs(effectBuffer, effectHit.path)
    const declaredEffectDiffuse = declaredEffectRefs
      .filter((value) => value.slot === 'diffuse')
      .map((value) => value.path)
    const declaredEffectNormal = declaredEffectRefs
      .filter((value) => value.slot === 'normal')
      .map((value) => value.path)

    const declaredEffectNormalHit = await resolveFirstExistingInOrder(declaredEffectNormal, lookup)

    const effectTextureHit = await resolveFirstExistingInOrder(declaredEffectDiffuse, lookup)
    if (!effectTextureHit) {
      console.log(`[Texture Debug] No effect texture found`)
      return null
    }
    console.log(`[Texture Debug] ✓ Resolved effect texture: ${effectTextureHit.path}`)
    if (declaredEffectNormalHit) console.log(`[Texture Debug] ✓ Resolved effect normal: ${declaredEffectNormalHit.path}`)

    return {
      texturePath: effectTextureHit.path,
      textureSourceLabel: effectTextureHit.source.sourceLabel,
      normalTexturePath: declaredNormalHit?.path ?? declaredEffectNormalHit?.path,
      normalTextureSourceLabel: declaredNormalHit?.source.sourceLabel ?? declaredEffectNormalHit?.source.sourceLabel,
      primaryUvSetIndex: 0,
      secondaryUvSetIndex: 0,
    }
  })()

  shaderTextureResolutionCache.set(cacheKey, pending)
  try {
    return await pending
  } catch (error) {
    shaderTextureResolutionCache.delete(cacheKey)
    throw error
  }
}

async function resolveTextureFromMeshShaderChain(
  meshRefs: string[],
  lookup: RepositoryLookup,
  options: ResolveTemplateOptions = {},
): Promise<{
  shaderPath?: string
  texturePath: string
  textureSourceLabel: string
  textureAddressU?: TextureAddressMode
  textureAddressV?: TextureAddressMode
  textureMipmapFilter?: TextureFilterMode
  textureMinificationFilter?: TextureFilterMode
  textureMagnificationFilter?: TextureFilterMode
  normalTexturePath?: string
  normalTextureSourceLabel?: string
  normalTextureAddressU?: TextureAddressMode
  normalTextureAddressV?: TextureAddressMode
} | null> {
  const normalizedMeshRefs = Array.from(
    new Set(meshRefs.map((value) => normalizeSwgPath(value)).filter((value) => ext(value) === '.msh')),
  )

  for (const meshRef of normalizedMeshRefs) {
    const meshSource = await lookup(meshRef)
    if (!meshSource) continue

    const meshBuffer = await meshSource.file.arrayBuffer()
    const meshShaderRefs = extractMshShaderRefs(meshBuffer, meshRef)
    for (const shaderRef of meshShaderRefs) {
      const textureHit = await resolveTextureFromShaderPath(shaderRef, lookup, options)
      if (!textureHit) continue

      const candidate = {
        shaderPath: shaderRef,
        texturePath: textureHit.texturePath,
        textureSourceLabel: textureHit.textureSourceLabel,
        textureAddressU: textureHit.textureAddressU,
        textureAddressV: textureHit.textureAddressV,
        textureMipmapFilter: textureHit.textureMipmapFilter,
        textureMinificationFilter: textureHit.textureMinificationFilter,
        textureMagnificationFilter: textureHit.textureMagnificationFilter,
        normalTexturePath: textureHit.normalTexturePath,
        normalTextureSourceLabel: textureHit.normalTextureSourceLabel,
        normalTextureAddressU: textureHit.normalTextureAddressU,
        normalTextureAddressV: textureHit.normalTextureAddressV,
      }
      return candidate
    }
  }

  return null
}

interface DecodedMeshHit {
  path: string
  sourceLabel: string
  mesh: PreviewMeshData
  meshParts?: PreviewMeshData[]
  meshPartTexturePaths?: string[]
  meshPartShaderPaths?: (string | undefined)[]
  meshPartTextureAddressU?: (TextureAddressMode | undefined)[]
  meshPartTextureAddressV?: (TextureAddressMode | undefined)[]
  meshPartTextureMipmapFilter?: (TextureFilterMode | undefined)[]
  meshPartTextureMinificationFilter?: (TextureFilterMode | undefined)[]
  meshPartTextureMagnificationFilter?: (TextureFilterMode | undefined)[]
  meshPartNormalTexturePaths?: (string | undefined)[]
  meshPartNormalTextureAddressU?: (TextureAddressMode | undefined)[]
  meshPartNormalTextureAddressV?: (TextureAddressMode | undefined)[]
  meshPartPrimaryUvSetIndices?: number[]
  meshPartPrimaryScaleU?: (number | undefined)[]
  meshPartPrimaryScaleV?: (number | undefined)[]
  meshPartSecondaryTexturePaths?: (string | undefined)[]
  meshPartSecondaryTextureAddressU?: (TextureAddressMode | undefined)[]
  meshPartSecondaryTextureAddressV?: (TextureAddressMode | undefined)[]
  meshPartSecondaryUvSetIndices?: number[]
  meshPartChunkTrace?: (string | undefined)[]
  meshPartEffectPaths?: (string | undefined)[]
}

interface ParsedInlineString {
  value: string
  nextOffset: number
}

interface PobCellAppearanceEntry {
  ref: string
  matrix?: number[]
}

function isReasonableAscii(text: string): boolean {
  return text.length > 0 && /^[\x20-\x7E]+$/.test(text)
}

function readInlineStringWithEncoding(
  bytes: Uint8Array,
  startOffset: number,
  encoding: 'u8' | 'u16le' | 'u16be' | 'u32le' | 'u32be' | 'cstr',
): ParsedInlineString {
  if (startOffset < 0 || startOffset >= bytes.length) {
    return { value: '', nextOffset: startOffset }
  }

  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  const readFixed = (len: number, prefix: number): ParsedInlineString => {
    if (len <= 0 || len > 4096 || startOffset + prefix + len > bytes.length) {
      return { value: '', nextOffset: startOffset }
    }
    const payload = bytes.subarray(startOffset + prefix, startOffset + prefix + len)
    const text = String.fromCharCode(...payload).replace(/\0+$/, '')
    if (!isReasonableAscii(text)) return { value: '', nextOffset: startOffset }
    return { value: text, nextOffset: startOffset + prefix + len }
  }

  if (encoding === 'u8') {
    return readFixed(bytes[startOffset], 1)
  }
  if (encoding === 'u16le' && startOffset + 2 <= bytes.length) {
    return readFixed(view.getUint16(startOffset, true), 2)
  }
  if (encoding === 'u16be' && startOffset + 2 <= bytes.length) {
    return readFixed(view.getUint16(startOffset, false), 2)
  }
  if (encoding === 'u32le' && startOffset + 4 <= bytes.length) {
    return readFixed(view.getUint32(startOffset, true), 4)
  }
  if (encoding === 'u32be' && startOffset + 4 <= bytes.length) {
    return readFixed(view.getUint32(startOffset, false), 4)
  }

  // c-string fallback.
  let end = startOffset
  while (end < bytes.length && bytes[end] !== 0) end += 1
  const text = String.fromCharCode(...bytes.subarray(startOffset, end))
  if (!isReasonableAscii(text)) return { value: '', nextOffset: startOffset }
  return { value: text, nextOffset: end < bytes.length ? end + 1 : end }
}

function readAffineMatrix(
  payload: Uint8Array,
  startOffset: number,
  littleEndian: boolean,
): number[] | undefined {
  if (startOffset < 0 || startOffset + 48 > payload.length) return undefined
  const view = new DataView(payload.buffer, payload.byteOffset, payload.byteLength)
  const out: number[] = []
  for (let i = 0; i < 12; i += 1) {
    out.push(view.getFloat32(startOffset + i * 4, littleEndian))
  }
  return out.every(Number.isFinite) ? out : undefined
}

function isReasonableAffineMatrix(matrix?: number[]): boolean {
  if (!matrix || matrix.length !== 12) return false

  const r0x = matrix[0], r0y = matrix[1], r0z = matrix[2]
  const r1x = matrix[4], r1y = matrix[5], r1z = matrix[6]
  const r2x = matrix[8], r2y = matrix[9], r2z = matrix[10]

  const linear = [r0x, r0y, r0z, r1x, r1y, r1z, r2x, r2y, r2z]
  if (linear.some((value) => !Number.isFinite(value) || Math.abs(value) > 4.5)) return false

  const len0 = Math.hypot(r0x, r0y, r0z)
  const len1 = Math.hypot(r1x, r1y, r1z)
  const len2 = Math.hypot(r2x, r2y, r2z)
  if (![len0, len1, len2].every((len) => Number.isFinite(len) && len > 0.2 && len < 4.0)) {
    return false
  }

  const dot01 = Math.abs(r0x * r1x + r0y * r1y + r0z * r1z)
  const dot02 = Math.abs(r0x * r2x + r0y * r2y + r0z * r2z)
  const dot12 = Math.abs(r1x * r2x + r1y * r2y + r1z * r2z)
  if (dot01 > 0.35 || dot02 > 0.35 || dot12 > 0.35) return false

  const det =
    r0x * (r1y * r2z - r1z * r2y)
    - r0y * (r1x * r2z - r1z * r2x)
    + r0z * (r1x * r2y - r1y * r2x)
  if (!Number.isFinite(det) || Math.abs(det) < 0.05 || Math.abs(det) > 12) return false

  const tx = matrix[3]
  const ty = matrix[7]
  const tz = matrix[11]
  if (![tx, ty, tz].every((value) => Number.isFinite(value) && Math.abs(value) < 25_000)) {
    return false
  }

  return true
}

function decodeCellDataTransformMatrix(dataPayload: Uint8Array, minOffset = 0): number[] | undefined {
  const candidateOffsets = Array.from(new Set([
    minOffset,
    minOffset + 4,
    minOffset + 8,
  ].filter((value) => value >= 0 && value + 48 <= dataPayload.length)))

  for (const offset of candidateOffsets) {
    const le = readAffineMatrix(dataPayload, offset, true)
    if (isReasonableAffineMatrix(le)) return le

    const be = readAffineMatrix(dataPayload, offset, false)
    if (isReasonableAffineMatrix(be)) return be
  }

  return undefined
}

function extractCellMeshRefFromDataChunk(
  dataPayload: Uint8Array,
  includeOutsideWorld = false,
): { meshRef: string; matrix?: number[] } | null {
  // Vanguard source parity for CELL DATA:
  // int numberOfPortals; byte unk; string name; string meshFile; byte floorFlag; [string floorFile]
  if (dataPayload.length < 8) return null

  const baseOffset = 5 // int32 + byte
  type Encoding = Parameters<typeof readInlineStringWithEncoding>[2]
  const encodings: Encoding[] = ['u8', 'u16le', 'u16be', 'u32le', 'u32be', 'cstr']

  for (const enc of encodings) {
    const nameRead = readInlineStringWithEncoding(dataPayload, baseOffset, enc)
    if (!nameRead.value) continue
    const meshRead = readInlineStringWithEncoding(dataPayload, nameRead.nextOffset, enc)
    if (!meshRead.value) continue

    const cellName = nameRead.value.trim().toLowerCase()
    const meshFile = meshRead.value.trim()
    if (!meshFile || !/\.(?:apt|msh|lod|cmp)$/i.test(meshFile)) continue

    // Interior assembly should skip outside/world cells; exterior traversal may include them.
    if (!includeOutsideWorld && /(^|[_/])(world|outside)([_/]|$)/i.test(cellName)) {
      continue
    }

    let matrixOffset = meshRead.nextOffset
    if (matrixOffset < dataPayload.length) {
      const floorFlag = dataPayload[matrixOffset]
      matrixOffset += 1
      if (floorFlag !== 0 && floorFlag !== 1) {
        continue
      }
      if (floorFlag === 1) {
        const floorRead = readInlineStringWithEncoding(dataPayload, matrixOffset, enc)
        if (!floorRead.value) continue
        matrixOffset = floorRead.nextOffset
      }
    }

    const matrix = decodeCellDataTransformMatrix(dataPayload, matrixOffset)

    return { meshRef: meshFile, matrix }
  }

  // Strict mode: if structured parse fails, do not guess.
  return null
}

function extractPobCellAppearanceRefs(
  buffer: ArrayBuffer,
  basePath: string,
  includeOutsideWorld = false,
): PobCellAppearanceEntry[] {
  const bytes = new Uint8Array(buffer)
  const root = parseBigEndianChunks(bytes)
  if (!root) return []

  const cells = flattenChunks(root).filter((chunk) => chunk.tag === 'FORM' && chunk.type === 'CELL')
  const entries: PobCellAppearanceEntry[] = []

  for (const cell of cells) {
    const dataChunk = flattenChunkSubtree(cell).find((chunk) => chunk.tag === 'DATA' && chunk.size >= 8)
    if (!dataChunk) continue
    const payload = bytes.subarray(dataChunk.dataStart, dataChunk.dataEnd)
    const parsed = extractCellMeshRefFromDataChunk(payload, includeOutsideWorld)
    if (!parsed) continue

    const selectedRef = resolveRefCandidates(basePath, parsed.meshRef)
      .find((value) => ['.apt', '.msh', '.lod', '.cmp'].includes(ext(value)))
    if (!selectedRef) continue

    entries.push({
      ref: selectedRef,
      matrix: parsed.matrix,
    })
  }

  return entries
}

/**
 * Extracts all HPNT (hardpoint) chunks from an IFF buffer (MSH/CMP/POB/etc.).
 * Each HPNT chunk's data layout is:
 *   float32[12] matrix     — 3x4 affine, row-major rotation+translation
 *   char[]      name       — null-terminated ASCII
 * Engine source uses little-endian floats; we fall back to big-endian if the
 * LE decode fails plausibility (some converters round-trip swapped).
 */
function extractHardpointsFromBuffer(buffer: ArrayBuffer, sourcePath?: string): MeshHardpoint[] {
  const bytes = new Uint8Array(buffer)
  const root = parseBigEndianChunks(bytes)
  if (!root) return []

  const out: MeshHardpoint[] = []
  const stack: BinaryChunk[] = [root]
  while (stack.length > 0) {
    const node = stack.pop()
    if (!node) continue
    for (const child of node.children) stack.push(child)
    if (node.tag !== 'HPNT' || node.type) continue
    const size = node.dataEnd - node.dataStart
    if (size < 49) continue // 12 floats + at least null terminator

    const payload = bytes.subarray(node.dataStart, node.dataEnd)
    let matrix = readAffineMatrix(payload, 0, true)
    if (!isReasonableAffineMatrix(matrix)) {
      const be = readAffineMatrix(payload, 0, false)
      if (isReasonableAffineMatrix(be)) matrix = be
    }
    if (!isReasonableAffineMatrix(matrix)) continue

    const nameRead = readInlineStringWithEncoding(payload, 48, 'cstr')
    const name = nameRead.value.trim()
    if (!name) continue

    out.push({ name, matrix: matrix as number[], sourcePath })
  }
  return out
}

function extractDtlLastChildRef(buffer: ArrayBuffer, basePath: string): string | null {
  const bytes = new Uint8Array(buffer)
  const root = parseBigEndianChunks(bytes)
  if (!root) return null

  const allForms = flattenChunks(root).filter((chunk) => chunk.tag === 'FORM')
  const dtla = allForms.find((chunk) => chunk.type === 'DTLA')
  if (!dtla) return null

  const versionForm = dtla.children.find((chunk) => chunk.tag === 'FORM')
  if (!versionForm) return null
  const dataForm = versionForm.children.find((chunk) => chunk.tag === 'FORM' && chunk.type === 'DATA')
  if (!dataForm) return null

  const chldChunks = dataForm.children.filter((chunk) => chunk.tag === 'CHLD' && chunk.size >= 8)
  if (chldChunks.length === 0) return null

  // Vanguard DetailAppearanceTemplate uses the last CHLD as firstMesh.
  for (let idx = chldChunks.length - 1; idx >= 0; idx -= 1) {
    const chld = chldChunks[idx]
    const payload = bytes.subarray(chld.dataStart, chld.dataEnd)
    if (payload.length < 8) continue

    const encodings: Array<'u8' | 'u16le' | 'u16be' | 'u32le' | 'u32be' | 'cstr'> = [
      'u8', 'u16le', 'u16be', 'u32le', 'u32be', 'cstr',
    ]

    let meshRef = ''
    for (const enc of encodings) {
      const parsed = readInlineStringWithEncoding(payload, 4, enc)
      if (!parsed.value) continue
      if (/\.(?:apt|msh|lod|cmp|pob)$/i.test(parsed.value.trim())) {
        meshRef = parsed.value.trim()
        break
      }
    }

    if (!meshRef) {
      const ascii = extractAsciiStrings(
        payload.buffer.slice(payload.byteOffset, payload.byteLength),
        4,
      )
      const fallback = ascii.find((value) => /\.(?:apt|msh|lod|cmp|pob)$/i.test(value))
      if (fallback) meshRef = fallback
    }

    if (!meshRef) continue
    const resolved = resolveRefCandidates(basePath, meshRef)
      .find((value) => ['.apt', '.msh', '.lod', '.cmp', '.pob'].includes(ext(value)))
    if (resolved) return resolved
  }

  return null
}

async function resolveMergedDecodedMeshes(
  refs: PobCellAppearanceEntry[],
  lookup: RepositoryLookup,
): Promise<DecodedMeshHit | null> {
  const enablePobCellTransforms = true
  const parts: PreviewMeshData[] = []
  const partTexturePaths: string[] = []
  const partTextureAddressU: (TextureAddressMode | undefined)[] = []
  const partTextureAddressV: (TextureAddressMode | undefined)[] = []
  const partTextureMipmapFilter: (TextureFilterMode | undefined)[] = []
  const partTextureMinificationFilter: (TextureFilterMode | undefined)[] = []
  const partTextureMagnificationFilter: (TextureFilterMode | undefined)[] = []
  const partNormalTexturePaths: (string | undefined)[] = []
  const partNormalTextureAddressU: (TextureAddressMode | undefined)[] = []
  const partNormalTextureAddressV: (TextureAddressMode | undefined)[] = []
  const partPrimaryUvSetIndices: number[] = []
  const partPrimaryScaleU: (number | undefined)[] = []
  const partPrimaryScaleV: (number | undefined)[] = []
  const partSecondaryTexturePaths: (string | undefined)[] = []
  const partSecondaryTextureAddressU: (TextureAddressMode | undefined)[] = []
  const partSecondaryTextureAddressV: (TextureAddressMode | undefined)[] = []
  const partSecondaryUvSetIndices: number[] = []
  const partChunkTrace: (string | undefined)[] = []
  const partShaderPaths: (string | undefined)[] = []
  const partEffectPaths: (string | undefined)[] = []
  const partPaths: string[] = []
  const labels: string[] = []
  const seen = new Set<string>()
  const seenResolvedMeshPaths = new Set<string>()

  const matrixSignature = (matrix?: number[]): string => {
    if (!matrix || matrix.length !== 12) return 'identity'
    return matrix.map((value) => Math.round(value * 10_000)).join(',')
  }

  for (const entry of refs) {
    const matrix = enablePobCellTransforms ? entry.matrix : undefined
    const norm = normalizeSwgPath(entry.ref)
    const instanceKey = `${norm}|${matrixSignature(matrix)}`
    if (seen.has(instanceKey)) continue
    seen.add(instanceKey)
    if (ext(norm) === '.msh') {
      const strictHit = await resolveMshPartMeshesWithTextures(norm, lookup)
      if (strictHit && strictHit.parts.length > 0) {
        for (let i = 0; i < strictHit.parts.length; i += 1) {
          parts.push(applyCellPartTransform(strictHit.parts[i], matrix))
          partTexturePaths.push(strictHit.texturePaths[i] ?? '')
          partTextureAddressU.push(strictHit.textureAddressU[i])
          partTextureAddressV.push(strictHit.textureAddressV[i])
          partTextureMipmapFilter.push(strictHit.textureMipmapFilter[i])
          partTextureMinificationFilter.push(strictHit.textureMinificationFilter[i])
          partTextureMagnificationFilter.push(strictHit.textureMagnificationFilter[i])
          partNormalTexturePaths.push(strictHit.normalTexturePaths[i])
          partNormalTextureAddressU.push(strictHit.normalTextureAddressU[i])
          partNormalTextureAddressV.push(strictHit.normalTextureAddressV[i])
          partPrimaryUvSetIndices.push(strictHit.primaryUvSetIndices[i] ?? 0)
          partPrimaryScaleU.push(strictHit.primaryScaleU[i])
          partPrimaryScaleV.push(strictHit.primaryScaleV[i])
          partSecondaryTexturePaths.push(strictHit.secondaryTexturePaths[i])
          partSecondaryTextureAddressU.push(strictHit.secondaryTextureAddressU[i])
          partSecondaryTextureAddressV.push(strictHit.secondaryTextureAddressV[i])
          partSecondaryUvSetIndices.push(strictHit.secondaryUvSetIndices[i] ?? 0)
          partChunkTrace.push(strictHit.chunkTrace[i])
          partShaderPaths.push(strictHit.shaderPaths[i])
          partEffectPaths.push(strictHit.effectPaths[i])
          partPaths.push(norm)
        }
        const src = await lookup(norm)
        if (src && !labels.includes(src.sourceLabel)) labels.push(src.sourceLabel)
        continue
      }
    }

    const hit = await resolveBestDecodedMesh([entry.ref], lookup, true, true)
    if (!hit) continue

    // If a non-MSH reference (APT/LOD/CMP/POB) resolved to an MSH path,
    // run strict MSH part extraction on that resolved path so per-part
    // shader/texture bindings are preserved instead of collapsing to a
    // single mesh-level fallback texture.
    if (ext(hit.path) === '.msh') {
      const resolvedStrictHit = await resolveMshPartMeshesWithTextures(hit.path, lookup)
      if (resolvedStrictHit && resolvedStrictHit.parts.length > 0) {
        for (let i = 0; i < resolvedStrictHit.parts.length; i += 1) {
          parts.push(applyCellPartTransform(resolvedStrictHit.parts[i], matrix))
          partTexturePaths.push(resolvedStrictHit.texturePaths[i] ?? '')
          partTextureAddressU.push(resolvedStrictHit.textureAddressU[i])
          partTextureAddressV.push(resolvedStrictHit.textureAddressV[i])
          partTextureMipmapFilter.push(resolvedStrictHit.textureMipmapFilter[i])
          partTextureMinificationFilter.push(resolvedStrictHit.textureMinificationFilter[i])
          partTextureMagnificationFilter.push(resolvedStrictHit.textureMagnificationFilter[i])
          partNormalTexturePaths.push(resolvedStrictHit.normalTexturePaths[i])
          partNormalTextureAddressU.push(resolvedStrictHit.normalTextureAddressU[i])
          partNormalTextureAddressV.push(resolvedStrictHit.normalTextureAddressV[i])
          partPrimaryUvSetIndices.push(resolvedStrictHit.primaryUvSetIndices[i] ?? 0)
          partPrimaryScaleU.push(resolvedStrictHit.primaryScaleU[i])
          partPrimaryScaleV.push(resolvedStrictHit.primaryScaleV[i])
          partSecondaryTexturePaths.push(resolvedStrictHit.secondaryTexturePaths[i])
          partSecondaryTextureAddressU.push(resolvedStrictHit.secondaryTextureAddressU[i])
          partSecondaryTextureAddressV.push(resolvedStrictHit.secondaryTextureAddressV[i])
          partSecondaryUvSetIndices.push(resolvedStrictHit.secondaryUvSetIndices[i] ?? 0)
          partChunkTrace.push(resolvedStrictHit.chunkTrace[i])
          partShaderPaths.push(resolvedStrictHit.shaderPaths[i])
          partEffectPaths.push(resolvedStrictHit.effectPaths[i])
          partPaths.push(hit.path)
        }
        if (!labels.includes(hit.sourceLabel)) labels.push(hit.sourceLabel)
        continue
      }
    }

    const resolvedPathKey = `${normalizeSwgPath(hit.path)}|${matrixSignature(matrix)}`
    if (seenResolvedMeshPaths.has(resolvedPathKey)) continue
    seenResolvedMeshPaths.add(resolvedPathKey)
    if (!hit.mesh || hit.mesh.positions.length < 9 || hit.mesh.indices.length < 3) continue

    if (hit.meshParts && hit.meshParts.length > 0) {
      for (let i = 0; i < hit.meshParts.length; i += 1) {
        const part = hit.meshParts[i]
        if (!part || part.positions.length < 9 || part.indices.length < 3) continue
        parts.push(applyCellPartTransform(part, matrix))
        partTexturePaths.push(hit.meshPartTexturePaths?.[i] ?? '')
        partTextureAddressU.push(hit.meshPartTextureAddressU?.[i])
        partTextureAddressV.push(hit.meshPartTextureAddressV?.[i])
        partTextureMipmapFilter.push(hit.meshPartTextureMipmapFilter?.[i])
        partTextureMinificationFilter.push(hit.meshPartTextureMinificationFilter?.[i])
        partTextureMagnificationFilter.push(hit.meshPartTextureMagnificationFilter?.[i])
        partNormalTexturePaths.push(hit.meshPartNormalTexturePaths?.[i])
        partNormalTextureAddressU.push(hit.meshPartNormalTextureAddressU?.[i])
        partNormalTextureAddressV.push(hit.meshPartNormalTextureAddressV?.[i])
        partPrimaryUvSetIndices.push(hit.meshPartPrimaryUvSetIndices?.[i] ?? 0)
        partPrimaryScaleU.push(hit.meshPartPrimaryScaleU?.[i])
        partPrimaryScaleV.push(hit.meshPartPrimaryScaleV?.[i])
        partSecondaryTexturePaths.push(hit.meshPartSecondaryTexturePaths?.[i])
        partSecondaryTextureAddressU.push(hit.meshPartSecondaryTextureAddressU?.[i])
        partSecondaryTextureAddressV.push(hit.meshPartSecondaryTextureAddressV?.[i])
        partSecondaryUvSetIndices.push(hit.meshPartSecondaryUvSetIndices?.[i] ?? 0)
        partChunkTrace.push(hit.meshPartChunkTrace?.[i])
        partShaderPaths.push(hit.meshPartShaderPaths?.[i])
        partEffectPaths.push(hit.meshPartEffectPaths?.[i])
        partPaths.push(hit.path)
      }
    } else {
      const meshOnlyTextureHit = await resolveTextureFromMeshShaderChain(
        [norm, hit.path],
        lookup,
        { strictDeclaredOnly: true },
      )
      parts.push(applyCellPartTransform(hit.mesh, matrix))
      partTexturePaths.push(hit.meshPartTexturePaths?.[0] ?? meshOnlyTextureHit?.texturePath ?? '')
      partTextureAddressU.push(hit.meshPartTextureAddressU?.[0] ?? meshOnlyTextureHit?.textureAddressU)
      partTextureAddressV.push(hit.meshPartTextureAddressV?.[0] ?? meshOnlyTextureHit?.textureAddressV)
      partTextureMipmapFilter.push(hit.meshPartTextureMipmapFilter?.[0] ?? meshOnlyTextureHit?.textureMipmapFilter)
      partTextureMinificationFilter.push(hit.meshPartTextureMinificationFilter?.[0] ?? meshOnlyTextureHit?.textureMinificationFilter)
      partTextureMagnificationFilter.push(hit.meshPartTextureMagnificationFilter?.[0] ?? meshOnlyTextureHit?.textureMagnificationFilter)
      partNormalTexturePaths.push(hit.meshPartNormalTexturePaths?.[0] ?? meshOnlyTextureHit?.normalTexturePath)
      partNormalTextureAddressU.push(hit.meshPartNormalTextureAddressU?.[0] ?? meshOnlyTextureHit?.normalTextureAddressU)
      partNormalTextureAddressV.push(hit.meshPartNormalTextureAddressV?.[0] ?? meshOnlyTextureHit?.normalTextureAddressV)
      partPrimaryUvSetIndices.push(hit.meshPartPrimaryUvSetIndices?.[0] ?? 0)
      partPrimaryScaleU.push(hit.meshPartPrimaryScaleU?.[0])
      partPrimaryScaleV.push(hit.meshPartPrimaryScaleV?.[0])
      partSecondaryTexturePaths.push(hit.meshPartSecondaryTexturePaths?.[0])
      partSecondaryTextureAddressU.push(hit.meshPartSecondaryTextureAddressU?.[0])
      partSecondaryTextureAddressV.push(hit.meshPartSecondaryTextureAddressV?.[0])
      partSecondaryUvSetIndices.push(hit.meshPartSecondaryUvSetIndices?.[0] ?? 0)
      partChunkTrace.push(hit.meshPartChunkTrace?.[0])
      partShaderPaths.push(hit.meshPartShaderPaths?.[0])
      partEffectPaths.push(hit.meshPartEffectPaths?.[0])
      partPaths.push(hit.path)
    }

    if (!labels.includes(hit.sourceLabel)) labels.push(hit.sourceLabel)
  }

  const merged = combineMeshes(parts)
  if (!merged) return null

  return {
    path: partPaths.length === 1 ? partPaths[0] : `merged:${partPaths[0] ?? 'interior'}`,
    sourceLabel: labels.join('+') || 'merged',
    mesh: merged,
    meshParts: parts,
    meshPartTexturePaths: partTexturePaths,
    meshPartTextureAddressU: partTextureAddressU,
    meshPartTextureAddressV: partTextureAddressV,
    meshPartTextureMipmapFilter: partTextureMipmapFilter,
    meshPartTextureMinificationFilter: partTextureMinificationFilter,
    meshPartTextureMagnificationFilter: partTextureMagnificationFilter,
    meshPartNormalTexturePaths: partNormalTexturePaths,
    meshPartNormalTextureAddressU: partNormalTextureAddressU,
    meshPartNormalTextureAddressV: partNormalTextureAddressV,
    meshPartPrimaryUvSetIndices: partPrimaryUvSetIndices,
    meshPartPrimaryScaleU: partPrimaryScaleU,
    meshPartPrimaryScaleV: partPrimaryScaleV,
    meshPartSecondaryTexturePaths: partSecondaryTexturePaths,
    meshPartSecondaryTextureAddressU: partSecondaryTextureAddressU,
    meshPartSecondaryTextureAddressV: partSecondaryTextureAddressV,
    meshPartSecondaryUvSetIndices: partSecondaryUvSetIndices,
    meshPartChunkTrace: partChunkTrace,
    meshPartShaderPaths: partShaderPaths,
  }
}

function meshQualityScore(mesh: PreviewMeshData): number {
  const verts = Math.floor(mesh.positions.length / 3)
  const tris = Math.floor(mesh.indices.length / 3)
  // Prefer candidate meshes with more triangles, then vertices.
  return tris * 1000 + verts
}

function meshPathDetailBias(path: string): number {
  const lower = normalizeSwgPath(path)
  let bias = 0

  // SWG mesh naming commonly uses r0/l0 for high-detail variants.
  if (/(^|[_/])r0([_/]|$)/.test(lower)) bias += 200_000
  if (/(^|[_/])l0([_/]|$)/.test(lower)) bias += 120_000
  if (/(^|[_/])r1([_/]|$)/.test(lower)) bias += 80_000
  if (/(^|[_/])l1([_/]|$)/.test(lower)) bias += 60_000

  // Penalize high-r variants which are typically reduced-detail proxies.
  const highR = lower.match(/(^|[_/])r(\d{1,2})([_/]|$)/)
  if (highR) {
    const level = Number.parseInt(highR[2], 10)
    if (Number.isFinite(level) && level >= 3) {
      bias -= Math.min(180_000, level * 8_000)
    }
  }

  return bias
}

interface MeshBounds {
  minX: number
  minY: number
  minZ: number
  maxX: number
  maxY: number
  maxZ: number
  sizeX: number
  sizeY: number
  sizeZ: number
}

function getMeshBounds(mesh: PreviewMeshData): MeshBounds | null {
  const vCount = Math.floor(mesh.positions.length / 3)
  if (vCount < 3) return null

  let minX = Number.POSITIVE_INFINITY
  let minY = Number.POSITIVE_INFINITY
  let minZ = Number.POSITIVE_INFINITY
  let maxX = Number.NEGATIVE_INFINITY
  let maxY = Number.NEGATIVE_INFINITY
  let maxZ = Number.NEGATIVE_INFINITY

  for (let i = 0; i < mesh.positions.length; i += 3) {
    const x = mesh.positions[i]
    const y = mesh.positions[i + 1]
    const z = mesh.positions[i + 2]
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue
    if (x < minX) minX = x
    if (y < minY) minY = y
    if (z < minZ) minZ = z
    if (x > maxX) maxX = x
    if (y > maxY) maxY = y
    if (z > maxZ) maxZ = z
  }

  if (!Number.isFinite(minX) || !Number.isFinite(maxX)) return null

  return {
    minX,
    minY,
    minZ,
    maxX,
    maxY,
    maxZ,
    sizeX: Math.max(0, maxX - minX),
    sizeY: Math.max(0, maxY - minY),
    sizeZ: Math.max(0, maxZ - minZ),
  }
}

function chooseInteriorPrimaryUvSet(mesh: PreviewMeshData, declaredPrimaryUvSet: number): number {
  const vertexCount = Math.floor(mesh.positions.length / 3)
  const getSet = (setIndex: number): number[] | undefined => {
    if (mesh.uvSets && setIndex >= 0 && setIndex < mesh.uvSets.length) {
      const fromUvSets = mesh.uvSets[setIndex]
      if (fromUvSets && fromUvSets.length >= vertexCount * 2) return fromUvSets
    }
    if (setIndex === 0 && mesh.uvs && mesh.uvs.length >= vertexCount * 2) return mesh.uvs
    if (setIndex === 1 && mesh.uvs1 && mesh.uvs1.length >= vertexCount * 2) return mesh.uvs1
    return undefined
  }

  const uvCoverage = (uvs?: number[]): number => {
    if (!uvs || uvs.length < 4) return 0
    let minU = Number.POSITIVE_INFINITY
    let minV = Number.POSITIVE_INFINITY
    let maxU = Number.NEGATIVE_INFINITY
    let maxV = Number.NEGATIVE_INFINITY
    let valid = 0
    for (let i = 0; i + 1 < uvs.length; i += 2) {
      const u = uvs[i]
      const v = uvs[i + 1]
      if (!Number.isFinite(u) || !Number.isFinite(v)) continue
      valid += 1
      if (u < minU) minU = u
      if (v < minV) minV = v
      if (u > maxU) maxU = u
      if (v > maxV) maxV = v
    }
    if (valid < 3) return 0
    return Math.max(0, maxU - minU) * Math.max(0, maxV - minV)
  }

  const declaredUv = getSet(declaredPrimaryUvSet)
  const declaredCoverage = uvCoverage(declaredUv)

  if (declaredUv) {
    // Keep strict declared channel unless it is effectively collapsed.
    if (declaredCoverage > 1e-7) return declaredPrimaryUvSet

    const uv0 = getSet(0)
    const uv1 = getSet(1)
    const uv0Coverage = uvCoverage(uv0)
    const uv1Coverage = uvCoverage(uv1)
    const bestFallbackCoverage = Math.max(uv0Coverage, uv1Coverage)

    // Only override when alternate UV set is clearly usable.
    if (bestFallbackCoverage > 1e-5) {
      if (uv1Coverage > uv0Coverage) return 1
      if (uv0Coverage > 0) return 0
    }

    return declaredPrimaryUvSet
  }

  const uv0 = getSet(0)
  const uv1 = getSet(1)
  if (uv0) return 0
  if (uv1) return 1
  return declaredPrimaryUvSet
}
function normalizeInteriorPartUvsForPreview(part: PreviewMeshData): PreviewMeshData {
  return {
    ...part,
    positions: part.positions.slice(),
    indices: part.indices.slice(),
    normals: part.normals ? part.normals.slice() : undefined,
    // Preserve authored UVs exactly for parity with in-game output.
    uvs: part.uvs ? part.uvs.slice() : undefined,
    uvs1: part.uvs1 ? part.uvs1.slice() : undefined,
    uvSets: part.uvSets ? part.uvSets.map((set) => set.slice()) : undefined,
  }
}

function sanitizeMesh(mesh: PreviewMeshData): PreviewMeshData | null {
  const vertexCount = Math.floor(mesh.positions.length / 3)
  if (vertexCount < 3 || mesh.indices.length < 3) return null

  const remap = new Int32Array(vertexCount)
  remap.fill(-1)

  const cleanPositions: number[] = []
  const hasUvs = Boolean(mesh.uvs && mesh.uvs.length >= vertexCount * 2)
  const hasUvs1 = Boolean(mesh.uvs1 && mesh.uvs1.length >= vertexCount * 2)
  const sourceUvSets = mesh.uvSets?.filter((set) => set.length >= vertexCount * 2) ?? []
  const cleanUvs: number[] = []
  const cleanUvs1: number[] = []
  const cleanUvSets: number[][] = sourceUvSets.map(() => [])

  for (let i = 0; i < vertexCount; i += 1) {
    const x = mesh.positions[i * 3]
    const y = mesh.positions[i * 3 + 1]
    const z = mesh.positions[i * 3 + 2]
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue
    remap[i] = Math.floor(cleanPositions.length / 3)
    cleanPositions.push(x, y, z)
    if (hasUvs) {
      cleanUvs.push(mesh.uvs![i * 2], mesh.uvs![i * 2 + 1])
    }
    if (hasUvs1) {
      cleanUvs1.push(mesh.uvs1![i * 2], mesh.uvs1![i * 2 + 1])
    }
    for (let setIndex = 0; setIndex < sourceUvSets.length; setIndex += 1) {
      cleanUvSets[setIndex].push(sourceUvSets[setIndex][i * 2], sourceUvSets[setIndex][i * 2 + 1])
    }
  }

  const cleanIndices: number[] = []
  for (let i = 0; i + 2 < mesh.indices.length; i += 3) {
    const a = mesh.indices[i]
    const b = mesh.indices[i + 1]
    const c = mesh.indices[i + 2]
    if (a < 0 || b < 0 || c < 0 || a >= vertexCount || b >= vertexCount || c >= vertexCount) continue
    if (a === b || b === c || a === c) continue

    const ra = remap[a]
    const rb = remap[b]
    const rc = remap[c]
    if (ra < 0 || rb < 0 || rc < 0) continue
    if (ra === rb || rb === rc || ra === rc) continue

    const ax = cleanPositions[ra * 3]
    const ay = cleanPositions[ra * 3 + 1]
    const az = cleanPositions[ra * 3 + 2]
    const bx = cleanPositions[rb * 3]
    const by = cleanPositions[rb * 3 + 1]
    const bz = cleanPositions[rb * 3 + 2]
    const cx = cleanPositions[rc * 3]
    const cy = cleanPositions[rc * 3 + 1]
    const cz = cleanPositions[rc * 3 + 2]

    if (!Number.isFinite(ax) || !Number.isFinite(ay) || !Number.isFinite(az) ||
        !Number.isFinite(bx) || !Number.isFinite(by) || !Number.isFinite(bz) ||
        !Number.isFinite(cx) || !Number.isFinite(cy) || !Number.isFinite(cz)) continue

    const area2 = triangleArea2(ax, ay, az, bx, by, bz, cx, cy, cz)
    if (!Number.isFinite(area2) || area2 < 1e-12) continue

    cleanIndices.push(ra, rb, rc)
  }

  if (cleanIndices.length < 3) return null

  const cleanNormals =
    mesh.normals && mesh.normals.length >= vertexCount * 3
      ? (() => {
          const outNormals: number[] = []
          for (let i = 0; i < vertexCount; i += 1) {
            if (remap[i] < 0) continue
            outNormals.push(mesh.normals![i * 3], mesh.normals![i * 3 + 1], mesh.normals![i * 3 + 2])
          }
          return outNormals
        })()
      : undefined

  return {
    positions: cleanPositions,
    indices: cleanIndices,
    normals: cleanNormals,
    uvs: hasUvs ? cleanUvs : undefined,
    uvs1: hasUvs1 ? cleanUvs1 : undefined,
    uvSets: cleanUvSets.length > 0 ? cleanUvSets : undefined,
    hasUvChannel: mesh.hasUvChannel,
    decodeDebug: mesh.decodeDebug,
  }
}

function normalizeCandidateMesh(mesh: PreviewMeshData): PreviewMeshData | null {
  const pruned = pruneSpikyTriangles(mesh) ?? mesh
  const cleaned = sanitizeMesh(pruned)
  if (!cleaned) return null

  const topoScore = scoreIndexTopology(cleaned.positions, cleaned.indices)
  if (!Number.isFinite(topoScore) || topoScore < 0) return null
  return cleaned
}

function combineMeshes(meshes: PreviewMeshData[]): PreviewMeshData | null {
  if (meshes.length === 0) return null
  if (meshes.length === 1) return normalizeCandidateMesh(meshes[0])

  // Pre-calculate total sizes to allocate buffers once
  const normalizedMeshes: (PreviewMeshData | null)[] = []
  let totalVertices = 0
  let totalIndices = 0
  let useUvs = false
  let useUvs1 = false
  let combinedHasUvChannel: boolean | undefined = undefined

  for (const mesh of meshes) {
    const normalized = normalizeCandidateMesh(mesh)
    if (!normalized) continue
    normalizedMeshes.push(normalized)
    totalVertices += Math.floor(normalized.positions.length / 3)
    totalIndices += normalized.indices.length
    if (normalized.uvs && normalized.uvs.length > 0) useUvs = true
    if (normalized.uvs1 && normalized.uvs1.length > 0) useUvs1 = true
    // Any part with a UV channel makes the combined mesh UV-capable
    if (normalized.hasUvChannel === true) combinedHasUvChannel = true
    else if (normalized.hasUvChannel === false && combinedHasUvChannel === undefined) combinedHasUvChannel = false
  }

  if (totalVertices === 0 || totalIndices === 0) return null

  // Pre-allocate typed arrays for better performance
  const positions = new Float32Array(totalVertices * 3)
  const indices = new Uint32Array(totalIndices)
  const uvs = useUvs ? new Float32Array(totalVertices * 2) : undefined
  const uvs1 = useUvs1 ? new Float32Array(totalVertices * 2) : undefined

  let posOffset = 0
  let indexOffset = 0
  let uvOffset = 0
  let uvs1Offset = 0

  for (const normalized of normalizedMeshes) {
    if (!normalized) continue

    const vertCount = Math.floor(normalized.positions.length / 3)
    const vertOffset = posOffset / 3

    // Copy positions as float32
    for (let i = 0; i < normalized.positions.length; i++) {
      positions[posOffset++] = normalized.positions[i]
    }

    // Copy UVs if needed
    if (useUvs && uvs) {
      if (normalized.uvs && normalized.uvs.length >= vertCount * 2) {
        for (let i = 0; i < vertCount * 2; i++) {
          uvs[uvOffset++] = normalized.uvs[i]
        }
      } else {
        for (let i = 0; i < vertCount * 2; i++) {
          uvs[uvOffset++] = 0
        }
      }
    }

    // Copy UV set 1 if needed
    if (useUvs1 && uvs1) {
      if (normalized.uvs1 && normalized.uvs1.length >= vertCount * 2) {
        for (let i = 0; i < vertCount * 2; i++) {
          uvs1[uvs1Offset++] = normalized.uvs1[i]
        }
      } else {
        for (let i = 0; i < vertCount * 2; i++) {
          uvs1[uvs1Offset++] = 0
        }
      }
    }

    // Copy indices with vertex offset
    for (let i = 0; i < normalized.indices.length; i++) {
      indices[indexOffset++] = normalized.indices[i] + vertOffset
    }
  }

  if (indexOffset < 3) return null

  // Only trim arrays if we over-allocated significantly
  const combined: PreviewMeshData = {
    positions: Array.from(positions.subarray(0, posOffset)),
    indices: Array.from(indices.subarray(0, indexOffset)),
    uvs: uvs ? Array.from(uvs.subarray(0, uvOffset)) : undefined,
    uvs1: uvs1 ? Array.from(uvs1.subarray(0, uvs1Offset)) : undefined,
    hasUvChannel: combinedHasUvChannel,
  }

  return normalizeCandidateMesh(combined)
}

function buildLodSiblingMeshCandidates(lodPath: string): string[] {
  const normalized = normalizeSwgPath(lodPath)
  if (!normalized.endsWith('.lod')) return []

  const stem = normalized.slice(0, -4)
  const out: string[] = []
  const seen = new Set<string>()
  const push = (value: string) => {
    const next = normalizeSwgPath(value)
    if (!next.endsWith('.msh') || seen.has(next)) return
    seen.add(next)
    out.push(next)
  }

  // Common SWG naming patterns for LOD-linked mesh files.
  push(`${stem}_l0_c0_l0.msh`)
  push(`${stem}_l0_c0_l1.msh`)
  push(`${stem}_c0_l0.msh`)
  push(`${stem}_c0_l1.msh`)
  push(`${stem}_l0.msh`)
  push(`${stem}_l1.msh`)
  push(`${stem}.msh`)

  return out
}

function extractLodMeshRefs(buffer: ArrayBuffer, basePath?: string): string[] {
  const refs = new Set<string>()
  const bytes = new Uint8Array(buffer)
  const base = basePath ? normalizeSwgPath(basePath) : ''

  const addRef = (raw: string) => {
    const trimmed = raw.trim()
    if (!trimmed) return
    const resolved = resolveRefCandidates(base, trimmed)
    for (const candidate of resolved) {
      if (ext(candidate) === '.msh') refs.add(candidate)
    }
  }

  try {
    const root = parseIff(buffer)
    const stack: IffChunk[] = [root]
    while (stack.length > 0) {
      const node = stack.pop()
      if (!node) continue
      for (const child of node.children) stack.push(child)

      if (!node.data || node.data.length === 0) continue

      if (node.tag === 'NAME') {
        const parsed = parseInlineSwgString(node.data)
        if (parsed) addRef(parsed)
      }

      // Some LOD variants store path-like strings in non-NAME leaves.
      const ascii = extractAsciiStrings(
        node.data.buffer.slice(node.data.byteOffset, node.data.byteOffset + node.data.byteLength),
        6,
      )
      for (const text of ascii) {
        if (/\.(?:msh)$/i.test(text)) addRef(text)
      }
    }
  } catch {
    // Fall back to raw buffer scan only if structured parse fails.
  }

  if (refs.size === 0) {
    const decoder = new TextDecoder('utf-8', { fatal: false })
    const fullText = decoder.decode(bytes)
    const meshMatches = fullText.match(/[a-z0-9_./\\-]+\.msh/gi)
    if (meshMatches) {
      for (const match of meshMatches) addRef(match)
    }
  }

  return Array.from(refs)
}


function parseTrailingLevel(path: string): number {
  const lower = normalizeSwgPath(path)
  const match = lower.match(/_l(\d+)\.msh$/)
  if (!match) return Number.POSITIVE_INFINITY
  return Number.parseInt(match[1], 10)
}

function chooseBestByLowestLevel(paths: string[]): string[] {
  const byGroup = new Map<string, string>()

  for (const raw of paths) {
    const next = normalizeSwgPath(raw)
    const groupKey = next.replace(/_l\d+\.msh$/, '.msh')
    const existing = byGroup.get(groupKey)
    if (!existing) {
      byGroup.set(groupKey, next)
      continue
    }

    const nextLevel = parseTrailingLevel(next)
    const existingLevel = parseTrailingLevel(existing)
    if (nextLevel < existingLevel || (nextLevel === existingLevel && next < existing)) {
      byGroup.set(groupKey, next)
    }
  }

  return Array.from(byGroup.values())
}

function choosePreferredSieMeshRefs(meshRefs: string[]): string[] {
  const normalized = Array.from(
    new Set(meshRefs.map((value) => normalizeSwgPath(value)).filter((value) => ext(value) === '.msh')),
  )
  if (normalized.length === 0) return []

  const componentRefs = normalized.filter((value) => /_c\d+_l\d+\.msh$/.test(value))
  if (componentRefs.length > 0) {
    const pickedComponent = chooseBestByLowestLevel(componentRefs)
    // SIE-style componentized assets should resolve from component families,
    // not unrelated fallback meshes.
    return Array.from(new Set(pickedComponent))
  }

  return chooseBestByLowestLevel(normalized)
}

interface CmpPartEntry {
  ref: string
  matrix?: number[]
}

function decodeCmpPartMatrix(payload: Uint8Array): number[] | undefined {
  if (payload.length < 48) return undefined
  const view = new DataView(payload.buffer, payload.byteOffset, payload.byteLength)
  const start = payload.length - 48
  const out: number[] = []
  for (let i = 0; i < 12; i += 1) {
    out.push(view.getFloat32(start + i * 4, true))
  }
  return out.every(Number.isFinite) ? out : undefined
}

function extractCmpPartEntries(buffer: ArrayBuffer, basePath: string): CmpPartEntry[] {
  const bytes = new Uint8Array(buffer)
  const root = parseBigEndianChunks(bytes)
  if (!root) return []

  const partChunks = flattenChunks(root).filter((chunk) => chunk.tag === 'PART' && chunk.size >= 8)
  const entries: CmpPartEntry[] = []

  for (const part of partChunks) {
    const payload = bytes.subarray(part.dataStart, part.dataEnd)
    const encodings: Array<'u8' | 'u16le' | 'u16be' | 'u32le' | 'u32be' | 'cstr'> = [
      'u8', 'u16le', 'u16be', 'u32le', 'u32be', 'cstr',
    ]

    let meshRef = ''
    for (const enc of encodings) {
      const parsed = readInlineStringWithEncoding(payload, 0, enc)
      if (!parsed.value) continue
      const candidate = parsed.value.trim()
      if (!/\.(?:apt|msh|lod|cmp|pob)$/i.test(candidate)) continue
      meshRef = candidate
      break
    }

    if (!meshRef) {
      const ascii = extractAsciiStrings(
        payload.buffer.slice(payload.byteOffset, payload.byteLength),
        4,
      )
      const fallback = ascii.find((value) => /\.(?:apt|msh|lod|cmp|pob)$/i.test(value))
      if (fallback) meshRef = fallback
    }

    if (!meshRef) continue
    const resolved = resolveRefCandidates(basePath, meshRef)
      .find((value) => ['.apt', '.msh', '.lod', '.cmp', '.pob'].includes(ext(value)))
    if (!resolved) continue

    entries.push({
      ref: resolved,
      matrix: decodeCmpPartMatrix(payload),
    })
  }

  return entries
}

function applyCmpPartTransform(mesh: PreviewMeshData, matrix?: number[]): PreviewMeshData {
  return applyAffinePartTransform(mesh, matrix, false)
}

function applyAffinePartTransform(
  mesh: PreviewMeshData,
  matrix: number[] | undefined,
  rowMajor: boolean,
): PreviewMeshData {
  if (!matrix || matrix.length !== 12) return mesh

  const out: PreviewMeshData = {
    positions: [],
    indices: mesh.indices.slice(),
    uvs: mesh.uvs ? mesh.uvs.slice() : undefined,
    uvs1: mesh.uvs1 ? mesh.uvs1.slice() : undefined,
    uvSets: mesh.uvSets ? mesh.uvSets.map((set) => set.slice()) : undefined,
    normals: mesh.normals ? [] : undefined,
    hasUvChannel: mesh.hasUvChannel,
  }

  const m00 = matrix[0]
  const m01 = rowMajor ? matrix[1] : matrix[4]
  const m02 = rowMajor ? matrix[2] : matrix[8]
  const m03 = rowMajor ? matrix[3] : matrix[3]
  const m10 = rowMajor ? matrix[4] : matrix[1]
  const m11 = matrix[5]
  const m12 = rowMajor ? matrix[6] : matrix[9]
  const m13 = rowMajor ? matrix[7] : matrix[7]
  const m20 = rowMajor ? matrix[8] : matrix[2]
  const m21 = rowMajor ? matrix[9] : matrix[6]
  const m22 = matrix[10]
  const m23 = rowMajor ? matrix[11] : matrix[11]

  for (let i = 0; i < mesh.positions.length; i += 3) {
    const x = mesh.positions[i]
    const y = mesh.positions[i + 1]
    const z = mesh.positions[i + 2]
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
      out.positions.push(x, y, z)
      if (out.normals && mesh.normals) {
        out.normals.push(mesh.normals[i] ?? 0, mesh.normals[i + 1] ?? 0, mesh.normals[i + 2] ?? 1)
      }
      continue
    }

    const tx = x * m00 + y * m01 + z * m02 + m03
    const ty = x * m10 + y * m11 + z * m12 + m13
    const tz = x * m20 + y * m21 + z * m22 + m23
    if (!Number.isFinite(tx) || !Number.isFinite(ty) || !Number.isFinite(tz)) {
      out.positions.push(x, y, z)
    } else {
      out.positions.push(tx, ty, tz)
    }

    if (out.normals && mesh.normals) {
      const nx = mesh.normals[i] ?? 0
      const ny = mesh.normals[i + 1] ?? 0
      const nz = mesh.normals[i + 2] ?? 1
      const tnx = nx * m00 + ny * m01 + nz * m02
      const tny = nx * m10 + ny * m11 + nz * m12
      const tnz = nx * m20 + ny * m21 + nz * m22
      const len = Math.hypot(tnx, tny, tnz)
      if (len > 1e-8 && Number.isFinite(len)) {
        out.normals.push(tnx / len, tny / len, tnz / len)
      } else {
        out.normals.push(nx, ny, nz)
      }
    }
  }

  return out
}

function scoreTransformedMeshPlausibility(original: PreviewMeshData, transformed: PreviewMeshData): number {
  const ob = getMeshBounds(original)
  const tb = getMeshBounds(transformed)
  if (!ob || !tb) return Number.POSITIVE_INFINITY

  const oExtent = Math.max(ob.sizeX, ob.sizeY, ob.sizeZ, 0.001)
  const tExtent = Math.max(tb.sizeX, tb.sizeY, tb.sizeZ, 0.001)
  const scaleRatio = tExtent / oExtent
  // CELL transforms should be rigid placement transforms, not heavy rescaling.
  if (!Number.isFinite(scaleRatio) || scaleRatio < 0.65 || scaleRatio > 1.6) {
    return Number.POSITIVE_INFINITY
  }

  const ocx = (ob.minX + ob.maxX) * 0.5
  const ocy = (ob.minY + ob.maxY) * 0.5
  const ocz = (ob.minZ + ob.maxZ) * 0.5
  const tcx = (tb.minX + tb.maxX) * 0.5
  const tcy = (tb.minY + tb.maxY) * 0.5
  const tcz = (tb.minZ + tb.maxZ) * 0.5
  const centerShift = Math.hypot(tcx - ocx, tcy - ocy, tcz - ocz)
  if (!Number.isFinite(centerShift) || centerShift > 400) {
    return Number.POSITIVE_INFINITY
  }

  const originalTopo = scoreIndexTopology(original.positions, original.indices)
  const transformedTopo = scoreIndexTopology(transformed.positions, transformed.indices)
  if (!Number.isFinite(transformedTopo) || transformedTopo < 0) {
    return Number.POSITIVE_INFINITY
  }
  if (Number.isFinite(originalTopo) && transformedTopo + 50_000 < originalTopo) {
    return Number.POSITIVE_INFINITY
  }

  return Math.abs(Math.log(scaleRatio)) + centerShift / 1200
}

function applyCellPartTransform(mesh: PreviewMeshData, matrix?: number[]): PreviewMeshData {
  if (!matrix || matrix.length !== 12) return mesh

  // CELL transform payloads vary across assets; choose the layout that yields
  // physically plausible geometry and reject pathological transforms.
  const cmpLayout = applyAffinePartTransform(mesh, matrix, false)
  const rowLayout = applyAffinePartTransform(mesh, matrix, true)
  const cmpScore = scoreTransformedMeshPlausibility(mesh, cmpLayout)
  const rowScore = scoreTransformedMeshPlausibility(mesh, rowLayout)

  if (!Number.isFinite(cmpScore) && !Number.isFinite(rowScore)) return mesh
  if (!Number.isFinite(cmpScore)) return rowLayout
  if (!Number.isFinite(rowScore)) return cmpLayout
  return rowScore < cmpScore ? rowLayout : cmpLayout
}

async function resolveCmpCompositeMesh(
  cmpPath: string,
  lookup: RepositoryLookup,
): Promise<DecodedMeshHit | null> {
  const normalizedCmp = normalizeSwgPath(cmpPath)
  const source = await lookup(normalizedCmp)
  if (!source) return null

  const buffer = await source.file.arrayBuffer()
  const entries = extractCmpPartEntries(buffer, normalizedCmp)
  if (entries.length === 0) return null

  const parts: PreviewMeshData[] = []
  const partPaths: string[] = []

  for (const entry of entries) {
    const hit = await resolveBestDecodedMesh([entry.ref], lookup, false, true)
    if (!hit?.mesh) continue
    const transformed = applyCmpPartTransform(hit.mesh, entry.matrix)
    const normalized = normalizeCandidateMesh(transformed)
    if (!normalized) continue
    parts.push(normalized)
    partPaths.push(hit.path)
  }

  const merged = combineMeshes(parts)
  if (!merged) return null

  return {
    path: partPaths.length === 1 ? partPaths[0] : `cmp-composite:${normalizedCmp}`,
    sourceLabel: source.sourceLabel,
    mesh: merged,
    meshParts: parts,
  }
}

async function collectSieChainMeshRefs(
  roots: string[],
  lookup: RepositoryLookup,
  maxDepth = 7,
): Promise<string[]> {
  const queue: Array<{ path: string; depth: number }> = roots
    .map((value) => normalizeSwgPath(value))
    .filter(Boolean)
    .map((path) => ({ path, depth: 0 }))
  const visited = new Set<string>()
  const meshes = new Set<string>()

  while (queue.length > 0) {
    const next = queue.shift()
    if (!next) break
    if (visited.has(next.path)) continue
    visited.add(next.path)

    const nextExt = ext(next.path)
    if (nextExt === '.msh') {
      meshes.add(next.path)
      continue
    }

    if (next.depth >= maxDepth) continue
    if (!['.pob', '.lod', '.cmp', '.apt'].includes(nextExt)) continue

    const source = await lookup(next.path)
    if (!source) continue

    const buffer = await source.file.arrayBuffer()
    const refs = getResolvedRefCandidatesCached(next.path, buffer)

    if (nextExt === '.lod') {
      // Parse LOD file to extract actual mesh references
      const lodRefs = extractLodMeshRefs(buffer, next.path)
      for (const lodRef of lodRefs) {
        refs.push(lodRef)
      }
      // Also try filename-based guessing as fallback
      for (const sibling of buildLodSiblingMeshCandidates(next.path)) {
        refs.push(sibling)
      }
    }

    for (const ref of refs) {
      const refExt = ext(ref)
      if (refExt === '.msh') {
        meshes.add(ref)
      } else if (['.pob', '.lod', '.cmp', '.apt'].includes(refExt)) {
        queue.push({ path: ref, depth: next.depth + 1 })
      }
    }
  }

  return Array.from(meshes)
}

async function collectDeclaredMeshRefs(
  roots: string[],
  lookup: RepositoryLookup,
  maxDepth = 7,
): Promise<string[]> {
  const queue: Array<{ path: string; depth: number }> = roots
    .map((value) => normalizeSwgPath(value))
    .filter(Boolean)
    .map((path) => ({ path, depth: 0 }))
  const visited = new Set<string>()
  const meshes = new Set<string>()

  while (queue.length > 0) {
    const next = queue.shift()
    if (!next) break
    if (visited.has(next.path)) continue
    visited.add(next.path)

    const nextExt = ext(next.path)
    if (nextExt === '.msh') {
      meshes.add(next.path)
      continue
    }

    if (next.depth >= maxDepth) continue
    if (!['.pob', '.apt', '.lod', '.cmp', '.iff'].includes(nextExt)) continue

    const source = await lookup(next.path)
    if (!source) continue

    const buffer = await source.file.arrayBuffer()
    const refs = getResolvedRefCandidatesCached(next.path, buffer)

    if (nextExt === '.apt') {
      const detailRef = extractDtlLastChildRef(buffer, next.path)
      if (detailRef) refs.push(detailRef)
    }

    if (nextExt === '.cmp') {
      const cmpEntries = extractCmpPartEntries(buffer, next.path)
      for (const entry of cmpEntries) refs.push(entry.ref)
    }

    if (nextExt === '.pob') {
      // Exterior chain must include world/outside CELL references for full shell/roof assembly.
      const pobCellRefs = extractPobCellAppearanceRefs(buffer, next.path, true)
      for (const entry of pobCellRefs) refs.push(entry.ref)
    }

    if (nextExt === '.lod') {
      // Deterministic: only explicit NAME references extracted from LOD content.
      const lodRefs = extractLodMeshRefs(buffer, next.path)
      for (const lodRef of lodRefs) refs.push(lodRef)
    }

    for (const ref of refs) {
      const refExt = ext(ref)
      if (refExt === '.msh') {
        meshes.add(ref)
      } else if (['.pob', '.apt', '.lod', '.cmp', '.iff'].includes(refExt)) {
        queue.push({ path: ref, depth: next.depth + 1 })
      }
    }
  }

  return Array.from(meshes)
}

async function resolveSieCompositeMesh(
  roots: string[],
  lookup: RepositoryLookup,
): Promise<DecodedMeshHit | null> {
  const chainRefs = await collectSieChainMeshRefs(roots, lookup)
  const preferredRefs = choosePreferredSieMeshRefs(chainRefs)
  if (preferredRefs.length === 0) return null

  const parts: PreviewMeshData[] = []
  const partTexturePaths: string[] = []
  const partTextureAddressU: (TextureAddressMode | undefined)[] = []
  const partTextureAddressV: (TextureAddressMode | undefined)[] = []
  const partTextureMipmapFilter: (TextureFilterMode | undefined)[] = []
  const partTextureMinificationFilter: (TextureFilterMode | undefined)[] = []
  const partTextureMagnificationFilter: (TextureFilterMode | undefined)[] = []
  const partNormalTexturePaths: (string | undefined)[] = []
  const partNormalTextureAddressU: (TextureAddressMode | undefined)[] = []
  const partNormalTextureAddressV: (TextureAddressMode | undefined)[] = []
  const partPrimaryUvSetIndices: number[] = []
  const partPrimaryScaleU: (number | undefined)[] = []
  const partPrimaryScaleV: (number | undefined)[] = []
  const partSecondaryTexturePaths: (string | undefined)[] = []
  const partSecondaryTextureAddressU: (TextureAddressMode | undefined)[] = []
  const partSecondaryTextureAddressV: (TextureAddressMode | undefined)[] = []
  const partSecondaryUvSetIndices: number[] = []
  const partChunkTrace: (string | undefined)[] = []
  const partShaderPaths: (string | undefined)[] = []
  const partEffectPaths: (string | undefined)[] = []
  let sourceLabel = ''

  for (const path of preferredRefs) {
    const source = await lookup(path)
    if (!source) continue
    if (!sourceLabel) sourceLabel = source.sourceLabel
    const partHit = await resolveMshPartMeshesWithTextures(path, lookup)
    if (partHit && partHit.parts.length > 0) {
      for (let i = 0; i < partHit.parts.length; i += 1) {
        parts.push(partHit.parts[i])
        partTexturePaths.push(partHit.texturePaths[i] ?? '')
        partTextureAddressU.push(partHit.textureAddressU[i])
        partTextureAddressV.push(partHit.textureAddressV[i])
        partTextureMipmapFilter.push(partHit.textureMipmapFilter[i])
        partTextureMinificationFilter.push(partHit.textureMinificationFilter[i])
        partTextureMagnificationFilter.push(partHit.textureMagnificationFilter[i])
        partNormalTexturePaths.push(partHit.normalTexturePaths[i])
        partNormalTextureAddressU.push(partHit.normalTextureAddressU[i])
        partNormalTextureAddressV.push(partHit.normalTextureAddressV[i])
        partPrimaryUvSetIndices.push(partHit.primaryUvSetIndices[i] ?? 0)
        partPrimaryScaleU.push(partHit.primaryScaleU[i])
        partPrimaryScaleV.push(partHit.primaryScaleV[i])
        partSecondaryTexturePaths.push(partHit.secondaryTexturePaths[i])
        partSecondaryTextureAddressU.push(partHit.secondaryTextureAddressU[i])
        partSecondaryTextureAddressV.push(partHit.secondaryTextureAddressV[i])
        partSecondaryUvSetIndices.push(partHit.secondaryUvSetIndices[i] ?? 0)
        partChunkTrace.push(partHit.chunkTrace[i])
        partShaderPaths.push(partHit.shaderPaths[i])
        partEffectPaths.push(partHit.effectPaths[i])
      }
      continue
    }

    const buffer = await source.file.arrayBuffer()
    const mesh = extractMeshFromAsset(buffer, path)
    if (!mesh) continue
    const cleaned = pruneSpikyTriangles(mesh) ?? mesh
    const topoScore = scoreIndexTopology(cleaned.positions, cleaned.indices)
    if (!Number.isFinite(topoScore) || topoScore < 0) continue
    parts.push(cleaned)

    const partTextureHit = await resolveTextureFromMeshShaderChain([path], lookup)
    partTexturePaths.push(partTextureHit?.texturePath ?? '')
    partTextureAddressU.push(partTextureHit?.textureAddressU)
    partTextureAddressV.push(partTextureHit?.textureAddressV)
    partTextureMipmapFilter.push(partTextureHit?.textureMipmapFilter)
    partTextureMinificationFilter.push(partTextureHit?.textureMinificationFilter)
    partTextureMagnificationFilter.push(partTextureHit?.textureMagnificationFilter)
    partNormalTexturePaths.push(undefined)
    partNormalTextureAddressU.push(partTextureHit?.normalTextureAddressU)
    partNormalTextureAddressV.push(partTextureHit?.normalTextureAddressV)
    partPrimaryUvSetIndices.push(0)
    partPrimaryScaleU.push(undefined)
    partPrimaryScaleV.push(undefined)
    partSecondaryTexturePaths.push(undefined)
    partSecondaryTextureAddressU.push(undefined)
    partSecondaryTextureAddressV.push(undefined)
    partSecondaryUvSetIndices.push(0)
    partChunkTrace.push(undefined)
    partShaderPaths.push(partTextureHit?.shaderPath)
    partEffectPaths.push(undefined)
  }

  const merged = combineMeshes(parts)
  if (!merged) return null

  const syntheticPath = `sie-composite:${normalizeSwgPath(roots[0] ?? 'unknown')}`
  return {
    path: syntheticPath,
    sourceLabel: sourceLabel || 'composite',
    mesh: merged,
    meshParts: parts,
    meshPartTexturePaths: partTexturePaths,
    meshPartTextureAddressU: partTextureAddressU,
    meshPartTextureAddressV: partTextureAddressV,
    meshPartTextureMipmapFilter: partTextureMipmapFilter,
    meshPartTextureMinificationFilter: partTextureMinificationFilter,
    meshPartTextureMagnificationFilter: partTextureMagnificationFilter,
    meshPartNormalTexturePaths: partNormalTexturePaths,
    meshPartNormalTextureAddressU: partNormalTextureAddressU,
    meshPartNormalTextureAddressV: partNormalTextureAddressV,
    meshPartPrimaryUvSetIndices: partPrimaryUvSetIndices,
    meshPartPrimaryScaleU: partPrimaryScaleU,
    meshPartPrimaryScaleV: partPrimaryScaleV,
    meshPartSecondaryTexturePaths: partSecondaryTexturePaths,
    meshPartSecondaryTextureAddressU: partSecondaryTextureAddressU,
    meshPartSecondaryTextureAddressV: partSecondaryTextureAddressV,
    meshPartSecondaryUvSetIndices: partSecondaryUvSetIndices,
    meshPartChunkTrace: partChunkTrace,
    meshPartShaderPaths: partShaderPaths,
    meshPartEffectPaths: partEffectPaths,
  }
}

async function resolveKnownPalaceComposite(
  appearancePath: string,
  lookup: RepositoryLookup,
): Promise<DecodedMeshHit | null> {
  const normalized = normalizeSwgPath(appearancePath)
  if (!normalized.includes('thm_nboo_thed_theed_palace')) return null

  const base = 'appearance/mesh/thm_nboo_thed_theed_palace_r0_mesh_l0'
  const parts: PreviewMeshData[] = []
  const partTexturePaths: string[] = []
  const partTextureAddressU: (TextureAddressMode | undefined)[] = []
  const partTextureAddressV: (TextureAddressMode | undefined)[] = []
  const partTextureMipmapFilter: (TextureFilterMode | undefined)[] = []
  const partTextureMinificationFilter: (TextureFilterMode | undefined)[] = []
  const partTextureMagnificationFilter: (TextureFilterMode | undefined)[] = []
  const partNormalTexturePaths: (string | undefined)[] = []
  const partNormalTextureAddressU: (TextureAddressMode | undefined)[] = []
  const partNormalTextureAddressV: (TextureAddressMode | undefined)[] = []
  const partPrimaryUvSetIndices: number[] = []
  const partPrimaryScaleU: (number | undefined)[] = []
  const partPrimaryScaleV: (number | undefined)[] = []
  const partSecondaryTexturePaths: (string | undefined)[] = []
  const partSecondaryTextureAddressU: (TextureAddressMode | undefined)[] = []
  const partSecondaryTextureAddressV: (TextureAddressMode | undefined)[] = []
  const partSecondaryUvSetIndices: number[] = []
  const partChunkTrace: (string | undefined)[] = []
  const partShaderPaths: (string | undefined)[] = []
  const partEffectPaths: (string | undefined)[] = []
  let sourceLabel = ''

  // SIE log shows this family is componentized as c0..c9 with l-level variants.
  for (let component = 0; component <= 9; component += 1) {
    let chosenPath: string | null = null

    for (let level = 0; level <= 5; level += 1) {
      const candidate = `${base}_c${component}_l${level}.msh`
      const hit = await lookup(candidate)
      if (!hit) continue
      chosenPath = candidate
      if (!sourceLabel) sourceLabel = hit.sourceLabel
      break
    }

    if (!chosenPath) continue

    const src = await lookup(chosenPath)
    if (!src) continue
    const partHit = await resolveMshPartMeshesWithTextures(chosenPath, lookup)
    if (partHit && partHit.parts.length > 0) {
      for (let i = 0; i < partHit.parts.length; i += 1) {
        parts.push(partHit.parts[i])
        partTexturePaths.push(partHit.texturePaths[i] ?? '')
        partTextureAddressU.push(partHit.textureAddressU[i])
        partTextureAddressV.push(partHit.textureAddressV[i])
        partTextureMipmapFilter.push(partHit.textureMipmapFilter[i])
        partTextureMinificationFilter.push(partHit.textureMinificationFilter[i])
        partTextureMagnificationFilter.push(partHit.textureMagnificationFilter[i])
        partNormalTexturePaths.push(partHit.normalTexturePaths[i])
        partNormalTextureAddressU.push(partHit.normalTextureAddressU[i])
        partNormalTextureAddressV.push(partHit.normalTextureAddressV[i])
        partPrimaryUvSetIndices.push(partHit.primaryUvSetIndices[i] ?? 0)
        partPrimaryScaleU.push(partHit.primaryScaleU[i])
        partPrimaryScaleV.push(partHit.primaryScaleV[i])
        partSecondaryTexturePaths.push(partHit.secondaryTexturePaths[i])
        partSecondaryTextureAddressU.push(partHit.secondaryTextureAddressU[i])
        partSecondaryTextureAddressV.push(partHit.secondaryTextureAddressV[i])
        partSecondaryUvSetIndices.push(partHit.secondaryUvSetIndices[i] ?? 0)
        partChunkTrace.push(partHit.chunkTrace[i])
        partShaderPaths.push(partHit.shaderPaths[i])
        partEffectPaths.push(partHit.effectPaths[i])
      }
      continue
    }

    const buffer = await src.file.arrayBuffer()
    const mesh = extractMeshFromAsset(buffer, chosenPath)
    if (!mesh) continue
    const cleaned = pruneSpikyTriangles(mesh) ?? mesh
    const topoScore = scoreIndexTopology(cleaned.positions, cleaned.indices)
    if (!Number.isFinite(topoScore) || topoScore < 0) continue
    parts.push(cleaned)

    const partTextureHit = await resolveTextureFromMeshShaderChain([chosenPath], lookup)
    partTexturePaths.push(partTextureHit?.texturePath ?? '')
    partTextureAddressU.push(partTextureHit?.textureAddressU)
    partTextureAddressV.push(partTextureHit?.textureAddressV)
    partTextureMipmapFilter.push(partTextureHit?.textureMipmapFilter)
    partTextureMinificationFilter.push(partTextureHit?.textureMinificationFilter)
    partTextureMagnificationFilter.push(partTextureHit?.textureMagnificationFilter)
    partNormalTexturePaths.push(undefined)
    partNormalTextureAddressU.push(partTextureHit?.normalTextureAddressU)
    partNormalTextureAddressV.push(partTextureHit?.normalTextureAddressV)
    partPrimaryUvSetIndices.push(0)
    partPrimaryScaleU.push(undefined)
    partPrimaryScaleV.push(undefined)
    partSecondaryTexturePaths.push(undefined)
    partSecondaryTextureAddressU.push(undefined)
    partSecondaryTextureAddressV.push(undefined)
    partSecondaryUvSetIndices.push(0)
    partChunkTrace.push(undefined)
    partShaderPaths.push(partTextureHit?.shaderPath)
    partEffectPaths.push(undefined)
  }

  const merged = combineMeshes(parts)
  if (!merged) return null

  return {
    path: 'sie-palace-composite:thm_nboo_thed_theed_palace',
    sourceLabel: sourceLabel || 'composite',
    mesh: merged,
    meshParts: parts,
    meshPartTexturePaths: partTexturePaths,
    meshPartTextureAddressU: partTextureAddressU,
    meshPartTextureAddressV: partTextureAddressV,
    meshPartTextureMipmapFilter: partTextureMipmapFilter,
    meshPartTextureMinificationFilter: partTextureMinificationFilter,
    meshPartTextureMagnificationFilter: partTextureMagnificationFilter,
    meshPartNormalTexturePaths: partNormalTexturePaths,
    meshPartNormalTextureAddressU: partNormalTextureAddressU,
    meshPartNormalTextureAddressV: partNormalTextureAddressV,
    meshPartPrimaryUvSetIndices: partPrimaryUvSetIndices,
    meshPartPrimaryScaleU: partPrimaryScaleU,
    meshPartPrimaryScaleV: partPrimaryScaleV,
    meshPartSecondaryTexturePaths: partSecondaryTexturePaths,
    meshPartSecondaryTextureAddressU: partSecondaryTextureAddressU,
    meshPartSecondaryTextureAddressV: partSecondaryTextureAddressV,
    meshPartSecondaryUvSetIndices: partSecondaryUvSetIndices,
    meshPartChunkTrace: partChunkTrace,
    meshPartShaderPaths: partShaderPaths,
    meshPartEffectPaths: partEffectPaths,
  }
}

// Retired heuristic helpers kept for reference while deterministic path is active.
void buildLikelyComponentCmpCandidates
void resolveSieCompositeMesh
void resolveKnownPalaceComposite

async function resolveBestDecodedMesh(
  refs: string[],
  lookup: RepositoryLookup,
  allowCmpComposite = true,
  deterministicOrder = false,
): Promise<DecodedMeshHit | null> {
  const candidates = Array.from(new Set(refs.map((value) => normalizeSwgPath(value))))
    .filter((value) => ext(value) === '.msh' || ext(value) === '.lod' || ext(value) === '.pob' || ext(value) === '.cmp')

  if (candidates.length === 0) return null

  let best: DecodedMeshHit | null = null
  let bestScore = Number.NEGATIVE_INFINITY
  const consider = (hit: DecodedMeshHit): DecodedMeshHit | null => {
    if (deterministicOrder) return hit
    const score = meshQualityScore(hit.mesh) + meshPathDetailBias(hit.path)
    if (!best || score > bestScore) {
      best = hit
      bestScore = score
    }
    return null
  }

  for (const candidate of candidates) {
    const source = await lookup(candidate)
    if (!source) continue

    const buffer = await source.file.arrayBuffer()
    const candidateExt = ext(candidate)

    if (candidateExt === '.cmp' && allowCmpComposite) {
      const cmpComposite = await resolveCmpCompositeMesh(candidate, lookup)
      if (cmpComposite?.mesh) {
        const chosen = consider(cmpComposite)
        if (chosen) return chosen
      }
      continue
    }

    if (candidateExt === '.lod') {
      // Parse LOD file to extract actual mesh references
      const lodRefs = extractLodMeshRefs(buffer, candidate)
      // Deterministic: include all explicit refs discoverable from the LOD file payload.
      const genericRefs = Array.from(
        new Set(getResolvedRefCandidatesCached(candidate, buffer)),
      ).filter((value) => ext(value) === '.msh')
      for (const ref of genericRefs) {
        if (!lodRefs.includes(ref)) lodRefs.push(ref)
      }

      if (!deterministicOrder) {
        // Fallback to filename-based guessing
        const synthesizedRefs = buildLodSiblingMeshCandidates(candidate)
        for (const sibling of synthesizedRefs) {
          if (!lodRefs.includes(sibling)) lodRefs.push(sibling)
        }
      }

      const lodPartMeshes: PreviewMeshData[] = []

      for (const lodRef of lodRefs) {
        const lodSource = await lookup(lodRef)
        if (!lodSource) continue
        const lodBuffer = await lodSource.file.arrayBuffer()
        const lodMesh = extractMeshFromAsset(lodBuffer, lodRef)
        if (!lodMesh) continue
        const normalizedLodMesh = normalizeCandidateMesh(lodMesh)
        if (!normalizedLodMesh) continue
        lodPartMeshes.push(normalizedLodMesh)
        const chosen = consider({
          path: lodRef,
          sourceLabel: lodSource.sourceLabel,
          mesh: normalizedLodMesh,
        })
        if (chosen) return chosen
      }

      const combinedLodMesh = combineMeshes(lodPartMeshes)
      const normalizedCombinedLodMesh = combinedLodMesh ? normalizeCandidateMesh(combinedLodMesh) : null
      if (normalizedCombinedLodMesh) {
        // Also consider a merged LOD candidate to preserve multi-part structures.
        const chosen = consider({
          path: candidate,
          sourceLabel: source.sourceLabel,
          mesh: normalizedCombinedLodMesh,
        })
        if (chosen) return chosen
      }

      // Deterministic mode: never decode raw .lod payload as geometry.
      // LOD must resolve through explicit mesh references only.
      continue
    }

    const decoded = extractMeshFromAsset(buffer, candidate)
    const normalizedDecoded = decoded ? normalizeCandidateMesh(decoded) : null
    if (normalizedDecoded) {
      const chosen = consider({
        path: candidate,
        sourceLabel: source.sourceLabel,
        mesh: normalizedDecoded,
      })
      if (chosen) return chosen
    }
  }

  return best
}

async function collectResolvedRefsRecursive(
  startPath: string,
  lookup: RepositoryLookup,
  maxDepth = 5,
): Promise<string[]> {
  const normalizedStart = normalizeSwgPath(startPath)
  const refs = new Set<string>()
  const visited = new Set<string>()
  const queue: Array<{ path: string; depth: number }> = [{ path: normalizedStart, depth: 0 }]

  while (queue.length > 0) {
    const next = queue.shift()
    if (!next) break
    if (visited.has(next.path)) continue
    visited.add(next.path)

    const hit = await lookup(next.path)
    if (!hit) continue

    const buffer = await hit.file.arrayBuffer()
    const localRefs = getResolvedRefCandidatesCached(next.path, buffer)

    for (const ref of localRefs) refs.add(ref)

    if (next.depth >= maxDepth) continue
    if (ext(next.path) !== '.iff') continue

    for (const ref of localRefs) {
      if (ext(ref) === '.iff' && !visited.has(ref)) {
        queue.push({ path: ref, depth: next.depth + 1 })
      }
    }
  }

  return Array.from(refs)
}

export async function resolveTemplateVisual(
  templatePath: string,
  lookup: RepositoryLookup,
  options: ResolveTemplateOptions = {},
): Promise<ResolvedTemplateVisual> {
  const normalizedTemplate = normalizeSwgPath(templatePath)
  const objectSource = await lookup(normalizedTemplate)
  if (!objectSource) {
    return {
      templatePath: normalizedTemplate,
      status: 'missing',
    }
  }

  const objectRefs = await collectResolvedRefsRecursive(normalizedTemplate, lookup)

  const appearanceHit = await resolveFirstExistingByExt(
    objectRefs,
    ['.pob', '.apt', '.lod', '.msh', '.sat', '.sht'],
    lookup,
  )
  const appearancePath = appearanceHit?.path

  if (!appearancePath) {
    return {
      templatePath: normalizedTemplate,
      objectPath: normalizedTemplate,
      sourceLabel: objectSource.sourceLabel,
      status: 'object-only',
    }
  }

  const appearanceSource = appearanceHit.source

  const appearanceBuffer = await appearanceSource.file.arrayBuffer()
  const appearanceRefs = getResolvedRefCandidatesCached(appearancePath, appearanceBuffer)
  const pobCellEntries = ext(appearancePath) === '.pob'
    ? extractPobCellAppearanceRefs(appearanceBuffer, appearancePath)
    : []

  const shaderHit = await resolveFirstExistingByExt(appearanceRefs, ['.sht', '.sat', '.trt'], lookup)
  let shaderPath = shaderHit?.path

  let meshCandidates = [...appearanceRefs]
  if (ext(appearancePath) === '.pob') {
    // Deterministic ordering: prefer explicitly referenced exterior LOD/MSH first.
    // Keep the top-level POB decode as fallback, not first choice.
    meshCandidates.push(appearancePath)
  }
  let detailPreferredRef: string | null = null
  if (ext(appearancePath) === '.apt') {
    detailPreferredRef = extractDtlLastChildRef(appearanceBuffer, appearancePath)
    if (detailPreferredRef) {
      meshCandidates = [detailPreferredRef, ...meshCandidates]
    }
  }
  if (ext(appearancePath) === '.msh' || ext(appearancePath) === '.lod') {
    meshCandidates.unshift(appearancePath)
  }

  const cmpRefs = meshCandidates.filter((p) => ext(p) === '.cmp')

  for (const cmp of cmpRefs) {
    const cmpMeshes = await collectCmpMeshRefs(cmp, lookup)
    meshCandidates.push(...cmpMeshes)
  }

  meshCandidates = Array.from(new Set(meshCandidates))
  const effectiveExteriorCandidatesUnordered = meshCandidates

  // ---------------- Hardpoint aggregation ----------------
  // Collect HPNT chunks along the entire appearance chain (top appearance buffer,
  // each POB cell's appearance, each declared CMP/MSH/LOD in the candidate set).
  // Dedupe by name + translation so we don't list mirror buffers twice.
  const collectedHardpoints: MeshHardpoint[] = []
  {
    const seenHpKeys = new Set<string>()
    const addHardpoints = (list: MeshHardpoint[]) => {
      for (const hp of list) {
        const t = hp.matrix
        const key = `${hp.name}|${t[3].toFixed(3)},${t[7].toFixed(3)},${t[11].toFixed(3)}`
        if (seenHpKeys.has(key)) continue
        seenHpKeys.add(key)
        collectedHardpoints.push(hp)
      }
    }
    addHardpoints(extractHardpointsFromBuffer(appearanceBuffer, appearancePath))

    const scanPaths = new Set<string>()
    for (const entry of pobCellEntries) scanPaths.add(entry.ref)
    for (const candidate of meshCandidates) {
      const e = ext(candidate)
      if (e === '.msh' || e === '.lod' || e === '.cmp' || e === '.pob' || e === '.apt') {
        scanPaths.add(candidate)
      }
    }
    scanPaths.delete(appearancePath)

    // Cap scans to avoid runaway lookups on pathological assets.
    let scanned = 0
    const MAX_HP_SCANS = 256
    for (const p of scanPaths) {
      if (scanned >= MAX_HP_SCANS) break
      scanned += 1
      try {
        const src = await lookup(p)
        if (!src) continue
        const buf = await src.file.arrayBuffer()
        addHardpoints(extractHardpointsFromBuffer(buf, p))
      } catch {
        // Best-effort: a single broken buffer should not block resolution.
      }
    }
  }
  // -------------------------------------------------------

  const extPriority = (value: string): number => {
    const valueExt = ext(value)
    if (valueExt === '.msh') return 0
    if (valueExt === '.lod') return 1
    if (valueExt === '.cmp') return 2
    if (valueExt === '.apt') return 3
    if (valueExt === '.pob') return 4
    return 5
  }
  const effectiveExteriorCandidates = [...effectiveExteriorCandidatesUnordered]
    .sort((a, b) => {
      const byType = extPriority(a) - extPriority(b)
      if (byType !== 0) return byType
      return a.localeCompare(b)
    })

  const declaredExteriorMeshRefs = await collectDeclaredMeshRefs(effectiveExteriorCandidates, lookup)
  const deterministicExteriorCandidates = declaredExteriorMeshRefs.length > 0
    ? declaredExteriorMeshRefs
    : effectiveExteriorCandidates

  console.log('[Resolver Exterior] Candidate trace', {
    template: normalizedTemplate,
    appearance: appearancePath,
    effectiveExteriorCandidates,
    declaredExteriorMeshRefs,
    deterministicExteriorCandidates,
  })

  const meshHintHit = await resolveFirstExistingByExt(deterministicExteriorCandidates, ['.msh', '.lod', '.pob'], lookup)
  const isPrimaryExteriorDecodeCandidate = (value: string): boolean => {
    const valueExt = ext(value)
    return valueExt === '.msh' || valueExt === '.lod' || valueExt === '.cmp' || valueExt === '.apt'
  }
  const exteriorDecodePrimary = deterministicExteriorCandidates.filter(isPrimaryExteriorDecodeCandidate)
  const exteriorEntriesPrimary: PobCellAppearanceEntry[] = exteriorDecodePrimary.map((ref) => ({ ref }))
  const mergedExteriorPrimaryHit = await resolveMergedDecodedMeshes(exteriorEntriesPrimary, lookup)
  const bestDecodedExteriorPrimaryHit = exteriorDecodePrimary.length > 0
    ? await resolveBestDecodedMesh(exteriorDecodePrimary, lookup, true, true)
    : null
  const sieCompositePrimaryHit = exteriorDecodePrimary.length > 0
    ? await resolveSieCompositeMesh(exteriorDecodePrimary, lookup)
    : null

  const secondaryCandidateSet = Array.from(new Set(effectiveExteriorCandidates.filter(isPrimaryExteriorDecodeCandidate)))
  const hasDifferentSecondarySet = secondaryCandidateSet.length !== exteriorDecodePrimary.length
    || secondaryCandidateSet.some((value, index) => value !== exteriorDecodePrimary[index])
  const exteriorEntriesSecondary: PobCellAppearanceEntry[] = hasDifferentSecondarySet
    ? secondaryCandidateSet.map((ref) => ({ ref }))
    : []
  const mergedExteriorSecondaryHit = exteriorEntriesSecondary.length > 0
    ? await resolveMergedDecodedMeshes(exteriorEntriesSecondary, lookup)
    : null
  const bestDecodedExteriorSecondaryHit = exteriorEntriesSecondary.length > 0
    ? await resolveBestDecodedMesh(secondaryCandidateSet, lookup, true, true)
    : null
  const sieCompositeSecondaryHit = exteriorEntriesSecondary.length > 0
    ? await resolveSieCompositeMesh(secondaryCandidateSet, lookup)
    : null
  const knownPalaceExteriorHit = await resolveKnownPalaceComposite(appearancePath, lookup)
  const extentOfHit = (hit: DecodedMeshHit | null): number | null => {
    if (!hit?.mesh) return null
    const bounds = getMeshBounds(hit.mesh)
    if (!bounds) return null
    const extent = Math.max(bounds.sizeX, bounds.sizeY, bounds.sizeZ)
    if (!Number.isFinite(extent) || extent <= 0) return null
    return extent
  }

  const rawExteriorCandidates = [
    { tag: 'knownPalace', hit: knownPalaceExteriorHit },
    { tag: 'siePrimary', hit: sieCompositePrimaryHit },
    { tag: 'mergedPrimary', hit: mergedExteriorPrimaryHit },
    { tag: 'bestPrimary', hit: bestDecodedExteriorPrimaryHit },
    { tag: 'sieSecondary', hit: sieCompositeSecondaryHit },
    { tag: 'mergedSecondary', hit: mergedExteriorSecondaryHit },
    { tag: 'bestSecondary', hit: bestDecodedExteriorSecondaryHit },
  ]

  const extentSamples = rawExteriorCandidates
    .map((entry) => extentOfHit(entry.hit))
    .filter((value): value is number => value !== null)
    .sort((a, b) => a - b)
  const referenceExtent = extentSamples.length > 0
    ? extentSamples[Math.floor(extentSamples.length / 2)]
    : null

  const exteriorHitScore = (hit: DecodedMeshHit | null): number => {
    if (!hit?.mesh) return Number.NEGATIVE_INFINITY
    const extent = extentOfHit(hit)
    if (extent === null) return Number.NEGATIVE_INFINITY

    if (referenceExtent !== null) {
      // Reject pathological exploded/collapsed geometry relative to peer candidates.
      if (extent > referenceExtent * 6 || extent < referenceExtent / 6) {
        return Number.NEGATIVE_INFINITY
      }
    }

    // Absolute safety gate for known bad decodes that explode to extreme bounds.
    if (extent > 5000) return Number.NEGATIVE_INFINITY

    const partCount = hit.meshParts?.length ?? 1
    const texturedPartCount = (hit.meshPartTexturePaths ?? []).filter((value) => Boolean(value && value.trim().length > 0)).length
    return partCount * 1_000_000_000 + texturedPartCount * 1_000_000 + meshQualityScore(hit.mesh)
  }

  const rankedExteriorHits = rawExteriorCandidates
    .map((entry) => ({
      ...entry,
      extent: extentOfHit(entry.hit),
      score: exteriorHitScore(entry.hit),
    }))
    .sort((a, b) => b.score - a.score)

  const isRenderableExterior = (entry: { score: number }) =>
    Number.isFinite(entry.score) && entry.score > Number.NEGATIVE_INFINITY

  const isPalaceFamily =
    normalizedTemplate.includes('palace_naboo_theed') ||
    appearancePath.includes('thm_nboo_thed_theed_palace') ||
    appearancePath.includes('shared_base_palace')

  const palacePrecedence = [
    'knownPalace',
    'siePrimary',
    'mergedPrimary',
    'bestPrimary',
    'sieSecondary',
    'mergedSecondary',
    'bestSecondary',
  ] as const

  const chosenExterior = isPalaceFamily
    ? palacePrecedence
        .map((tag) => rankedExteriorHits.find((entry) => entry.tag === tag))
        .find((entry): entry is NonNullable<typeof entry> => Boolean(entry && isRenderableExterior(entry)))
      ?? rankedExteriorHits.find((entry) => isRenderableExterior(entry))
    : rankedExteriorHits.find((entry) => isRenderableExterior(entry))

  const decodedMeshHit = chosenExterior?.hit ?? null
  const meshPath = decodedMeshHit?.path ?? meshHintHit?.path

  console.log('[Resolver Exterior] Merge result', {
    template: normalizedTemplate,
    meshHintPath: meshHintHit?.path,
    exteriorDecodePrimary,
    secondaryCandidateSet,
    mergedPrimaryPath: mergedExteriorPrimaryHit?.path,
    mergedPrimaryPartCount: mergedExteriorPrimaryHit?.meshParts?.length ?? 0,
    bestPrimaryPath: bestDecodedExteriorPrimaryHit?.path,
    bestPrimaryPartCount: bestDecodedExteriorPrimaryHit?.meshParts?.length ?? (bestDecodedExteriorPrimaryHit?.mesh ? 1 : 0),
    siePrimaryPath: sieCompositePrimaryHit?.path,
    siePrimaryPartCount: sieCompositePrimaryHit?.meshParts?.length ?? (sieCompositePrimaryHit?.mesh ? 1 : 0),
    mergedSecondaryPath: mergedExteriorSecondaryHit?.path,
    mergedSecondaryPartCount: mergedExteriorSecondaryHit?.meshParts?.length ?? 0,
    bestSecondaryPath: bestDecodedExteriorSecondaryHit?.path,
    bestSecondaryPartCount: bestDecodedExteriorSecondaryHit?.meshParts?.length ?? (bestDecodedExteriorSecondaryHit?.mesh ? 1 : 0),
    sieSecondaryPath: sieCompositeSecondaryHit?.path,
    sieSecondaryPartCount: sieCompositeSecondaryHit?.meshParts?.length ?? (sieCompositeSecondaryHit?.mesh ? 1 : 0),
    knownPalacePath: knownPalaceExteriorHit?.path,
    knownPalacePartCount: knownPalaceExteriorHit?.meshParts?.length ?? (knownPalaceExteriorHit?.mesh ? 1 : 0),
    isPalaceFamily,
    palacePrecedence,
    referenceExtent,
    chosenExteriorTag: chosenExterior?.tag,
    rankedExteriorHits: rankedExteriorHits.map((entry) => ({ tag: entry.tag, score: entry.score, extent: entry.extent, path: entry.hit?.path })),
    decodedPath: decodedMeshHit?.path,
    decodedPartCount: decodedMeshHit?.meshParts?.length ?? (decodedMeshHit?.mesh ? 1 : 0),
    finalMeshPath: meshPath,
  })

  let texturePath: string | undefined
  let textureAddressU: TextureAddressMode | undefined
  let textureAddressV: TextureAddressMode | undefined
  let textureMipmapFilter: TextureFilterMode | undefined
  let textureMinificationFilter: TextureFilterMode | undefined
  let textureMagnificationFilter: TextureFilterMode | undefined
  let normalTexturePath: string | undefined
  let normalTextureAddressU: TextureAddressMode | undefined
  let normalTextureAddressV: TextureAddressMode | undefined
  let textureSourceLabel: string | undefined
  let normalTextureSourceLabel: string | undefined
  let effectPath: string | undefined
  let effectOptionCodes: string[] | undefined
  let vertexProgramPaths: string[] | undefined
  let pixelProgramPaths: string[] | undefined
  let shaderStageTexturePaths: string[] | undefined
  let shaderStageNormalTexturePaths: string[] | undefined
  if (shaderHit) {
    const shaderBuffer = await shaderHit.source.file.arrayBuffer()
    const stageRefs = extractShaderStageTextureRefs(shaderBuffer, shaderHit.path)
    shaderStageTexturePaths = stageRefs.stageTextures
    shaderStageNormalTexturePaths = stageRefs.stageNormals
    const shaderRefs = getResolvedRefCandidatesCached(shaderHit.path, shaderBuffer)
    const effectHit = await resolveFirstExistingByExt(shaderRefs, ['.eft'], lookup)
    if (effectHit) {
      effectPath = effectHit.path
      const effectBuffer = await effectHit.source.file.arrayBuffer()
      const recipe = extractEffectProgramAndOptionRefs(effectBuffer, effectHit.path)
      effectOptionCodes = recipe.optionCodes
      vertexProgramPaths = recipe.vertexPrograms
      pixelProgramPaths = recipe.pixelPrograms
    }

    const textureHit = await resolveTextureFromShaderPath(shaderHit.path, lookup, options)

    if (textureHit) {
      texturePath = textureHit.texturePath
      textureAddressU = textureHit.textureAddressU
      textureAddressV = textureHit.textureAddressV
      textureMipmapFilter = textureHit.textureMipmapFilter
      textureMinificationFilter = textureHit.textureMinificationFilter
      textureMagnificationFilter = textureHit.textureMagnificationFilter
      normalTexturePath = textureHit.normalTexturePath
      normalTextureAddressU = textureHit.normalTextureAddressU
      normalTextureAddressV = textureHit.normalTextureAddressV
      textureSourceLabel = textureHit.textureSourceLabel
      normalTextureSourceLabel = textureHit.normalTextureSourceLabel
    }
  }

  // No fallback texture guessing in deterministic mode.

  let pobMeshFallback: PreviewMeshData | undefined
  let pobInteriorHit: DecodedMeshHit | null = null
  if (ext(appearancePath) === '.pob') {
    // Keep embedded POB geometry as a fallback, but still prefer external LOD/MSH meshes.
    pobMeshFallback = extractMeshFromAsset(appearanceBuffer, appearancePath) ?? undefined
    if (pobCellEntries.length > 0) {
      pobInteriorHit = await resolveMergedDecodedMeshes(pobCellEntries, lookup)
    }
  }

  if (!meshPath) {
    if (pobMeshFallback) {
      return {
        templatePath: normalizedTemplate,
        objectPath: normalizedTemplate,
        appearancePath,
        shaderPath,
        texturePath,
        textureAddressU,
        textureAddressV,
        normalTexturePath,
        normalTextureAddressU,
        normalTextureAddressV,
        textureSourceLabel,
        normalTextureSourceLabel,
        effectPath,
        effectOptionCodes,
        vertexProgramPaths,
        pixelProgramPaths,
        shaderStageTexturePaths,
        shaderStageNormalTexturePaths,
        meshPath: appearancePath,
        sourceLabel: appearanceSource.sourceLabel,
        mesh: pobMeshFallback,
        hardpoints: collectedHardpoints.length ? collectedHardpoints : undefined,
        status: 'mesh',
      }
    }

    return {
      templatePath: normalizedTemplate,
      objectPath: normalizedTemplate,
      appearancePath,
      shaderPath,
      texturePath,
      textureAddressU,
      textureAddressV,
      textureMipmapFilter,
      textureMinificationFilter,
      textureMagnificationFilter,
      normalTexturePath,
      normalTextureAddressU,
      normalTextureAddressV,
      textureSourceLabel,
      normalTextureSourceLabel,
      effectPath,
      effectOptionCodes,
      vertexProgramPaths,
      pixelProgramPaths,
      shaderStageTexturePaths,
      shaderStageNormalTexturePaths,
      sourceLabel: appearanceSource.sourceLabel,
      hardpoints: collectedHardpoints.length ? collectedHardpoints : undefined,
      status: 'appearance-only',
    }
  }

  const finalMeshParts: PreviewMeshData[] = []
  const finalPathParts: string[] = []
  const finalSourceParts: string[] = []
  const renderMeshParts: PreviewMeshData[] = []
  const renderMeshPartTexturePaths: string[] = []
  const renderMeshPartTextureAddressU: (TextureAddressMode | undefined)[] = []
  const renderMeshPartTextureAddressV: (TextureAddressMode | undefined)[] = []
  const renderMeshPartTextureMipmapFilter: (TextureFilterMode | undefined)[] = []
  const renderMeshPartTextureMinificationFilter: (TextureFilterMode | undefined)[] = []
  const renderMeshPartTextureMagnificationFilter: (TextureFilterMode | undefined)[] = []
  const renderMeshPartNormalTexturePaths: (string | undefined)[] = []
  const renderMeshPartNormalTextureAddressU: (TextureAddressMode | undefined)[] = []
  const renderMeshPartNormalTextureAddressV: (TextureAddressMode | undefined)[] = []
  const renderMeshPartPrimaryUvSetIndices: number[] = []
  const renderMeshPartSecondaryTexturePaths: (string | undefined)[] = []
  const renderMeshPartSecondaryTextureAddressU: (TextureAddressMode | undefined)[] = []
  const renderMeshPartSecondaryTextureAddressV: (TextureAddressMode | undefined)[] = []
  const renderMeshPartSecondaryUvSetIndices: number[] = []
  const renderMeshPartChunkTrace: (string | undefined)[] = []
  const renderMeshPartShaderPaths: (string | undefined)[] = []
  const renderMeshPartEffectPaths: (string | undefined)[] = []
  const renderMeshPartDomains: Array<'exterior' | 'interior'> = []
  const seenRenderPartKeys = new Set<string>()

  const buildRenderPartKey = (
    part: PreviewMeshData,
    texturePath: string,
    primaryUvSet: number,
  ): string => {
    const bounds = getMeshBounds(part)
    const triCount = Math.floor(part.indices.length / 3)
    const vertCount = Math.floor(part.positions.length / 3)
    if (bounds) {
      const q = (value: number) => Math.round(value * 100)
      const cx = (bounds.minX + bounds.maxX) * 0.5
      const cy = (bounds.minY + bounds.maxY) * 0.5
      const cz = (bounds.minZ + bounds.maxZ) * 0.5
      return [
        texturePath,
        primaryUvSet,
        triCount,
        vertCount,
        q(cx), q(cy), q(cz),
        q(bounds.sizeX), q(bounds.sizeY), q(bounds.sizeZ),
      ].join('|')
    }

    const sample = [
      part.positions[0] ?? 0,
      part.positions[1] ?? 0,
      part.positions[2] ?? 0,
      part.positions[3] ?? 0,
      part.positions[4] ?? 0,
      part.positions[5] ?? 0,
    ].map((value) => Math.round(value * 1000))
    return [texturePath, primaryUvSet, triCount, vertCount, ...sample].join('|')
  }

  const processMeshParts = (
    parts: PreviewMeshData[],
    texturePaths: string[] = [],
    textureAddressU: (TextureAddressMode | undefined)[] = [],
    textureAddressV: (TextureAddressMode | undefined)[] = [],
    textureMipmapFilter: (TextureFilterMode | undefined)[] = [],
    textureMinificationFilter: (TextureFilterMode | undefined)[] = [],
    textureMagnificationFilter: (TextureFilterMode | undefined)[] = [],
    normalTexturePaths: (string | undefined)[] = [],
    normalTextureAddressU: (TextureAddressMode | undefined)[] = [],
    normalTextureAddressV: (TextureAddressMode | undefined)[] = [],
    primaryUvSetIndices: number[] = [],
    secondaryTexturePaths: (string | undefined)[] = [],
    secondaryTextureAddressU: (TextureAddressMode | undefined)[] = [],
    secondaryTextureAddressV: (TextureAddressMode | undefined)[] = [],
    secondaryUvSetIndices: number[] = [],
    shaderPaths: (string | undefined)[] = [],
    chunkTrace: (string | undefined)[] = [],
    effectPaths: (string | undefined)[] = [],
    domain: 'exterior' | 'interior' = 'exterior',
  ) => {
    for (let i = 0; i < parts.length; i++) {
      const part = parts[i]
      const baseTexture = texturePaths[i] ?? ''
      const normalTexture = normalTexturePaths[i] ?? ''
      const secondaryTexture = secondaryTexturePaths[i] ?? ''
      const trace = chunkTrace[i] ?? ''

      const vertexCount = Math.floor(part.positions.length / 3)
      const hasUv0 = Boolean(part.uvs && part.uvs.length >= vertexCount * 2)
      const hasUv1 = Boolean(part.uvs1 && part.uvs1.length >= vertexCount * 2)
      const hasUvSet = Boolean(
        part.uvSets && part.uvSets.some((set) => set.length >= vertexCount * 2),
      )
      const hasAnyUv = hasUv0 || hasUv1 || hasUvSet

      // Collision/aux parts often appear in decoded meshes with no shader binding and no UVs.
      // Apply this cull only to interior assembly; exterior shells may include valid geometry
      // chunks without explicit per-part bindings.
      const hasBinding = Boolean(baseTexture || normalTexture || secondaryTexture || (trace && trace !== 'n/a'))
      if (domain === 'interior' && !hasBinding && !hasAnyUv) continue

      const primaryUv = primaryUvSetIndices[i] ?? 0
      const partKey = buildRenderPartKey(part, baseTexture, primaryUv)
      if (seenRenderPartKeys.has(partKey)) continue
      seenRenderPartKeys.add(partKey)

      renderMeshParts.push(part)
      renderMeshPartTexturePaths.push(baseTexture)
      renderMeshPartTextureAddressU.push(textureAddressU[i])
      renderMeshPartTextureAddressV.push(textureAddressV[i])
      renderMeshPartTextureMipmapFilter.push(textureMipmapFilter[i])
      renderMeshPartTextureMinificationFilter.push(textureMinificationFilter[i])
      renderMeshPartTextureMagnificationFilter.push(textureMagnificationFilter[i])
      renderMeshPartNormalTexturePaths.push(normalTexturePaths[i])
      renderMeshPartNormalTextureAddressU.push(normalTextureAddressU[i])
      renderMeshPartNormalTextureAddressV.push(normalTextureAddressV[i])
      renderMeshPartPrimaryUvSetIndices.push(primaryUv)
      renderMeshPartSecondaryTexturePaths.push(secondaryTexturePaths[i])
      renderMeshPartSecondaryTextureAddressU.push(secondaryTextureAddressU[i])
      renderMeshPartSecondaryTextureAddressV.push(secondaryTextureAddressV[i])
      renderMeshPartSecondaryUvSetIndices.push(secondaryUvSetIndices[i] ?? 0)
      renderMeshPartShaderPaths.push(shaderPaths[i])
      renderMeshPartEffectPaths.push(effectPaths[i])
      renderMeshPartChunkTrace.push(chunkTrace[i])
      renderMeshPartDomains.push(domain)
    }
  };

  if (decodedMeshHit?.mesh) {
    finalMeshParts.push(decodedMeshHit.mesh)
    finalPathParts.push(decodedMeshHit.path)
    finalSourceParts.push(decodedMeshHit.sourceLabel)

    const exteriorParts = decodedMeshHit.meshParts?.length
      ? decodedMeshHit.meshParts
      : [decodedMeshHit.mesh]
    processMeshParts(
      exteriorParts,
      decodedMeshHit.meshPartTexturePaths,
      decodedMeshHit.meshPartTextureAddressU,
      decodedMeshHit.meshPartTextureAddressV,
      decodedMeshHit.meshPartTextureMipmapFilter,
      decodedMeshHit.meshPartTextureMinificationFilter,
      decodedMeshHit.meshPartTextureMagnificationFilter,
      decodedMeshHit.meshPartNormalTexturePaths,
      decodedMeshHit.meshPartNormalTextureAddressU,
      decodedMeshHit.meshPartNormalTextureAddressV,
      decodedMeshHit.meshPartPrimaryUvSetIndices,
      decodedMeshHit.meshPartSecondaryTexturePaths,
      decodedMeshHit.meshPartSecondaryTextureAddressU,
      decodedMeshHit.meshPartSecondaryTextureAddressV,
      decodedMeshHit.meshPartSecondaryUvSetIndices,
      decodedMeshHit.meshPartShaderPaths,
      decodedMeshHit.meshPartChunkTrace,
      decodedMeshHit.meshPartEffectPaths,
      'exterior',
    )
  }

  if (pobInteriorHit?.mesh) {
    let interiorParts = pobInteriorHit.meshParts?.length
      ? pobInteriorHit.meshParts
      : [pobInteriorHit.mesh]
    let interiorTextures = pobInteriorHit.meshPartTexturePaths || []
    let interiorTextureAddressU: (TextureAddressMode | undefined)[] = pobInteriorHit.meshPartTextureAddressU || []
    let interiorTextureAddressV: (TextureAddressMode | undefined)[] = pobInteriorHit.meshPartTextureAddressV || []
    let interiorTextureMipmapFilter: (TextureFilterMode | undefined)[] = pobInteriorHit.meshPartTextureMipmapFilter || []
    let interiorTextureMinificationFilter: (TextureFilterMode | undefined)[] = pobInteriorHit.meshPartTextureMinificationFilter || []
    let interiorTextureMagnificationFilter: (TextureFilterMode | undefined)[] = pobInteriorHit.meshPartTextureMagnificationFilter || []
    let interiorNormals: (string | undefined)[] = pobInteriorHit.meshPartNormalTexturePaths || []
    let interiorNormalAddressU: (TextureAddressMode | undefined)[] = pobInteriorHit.meshPartNormalTextureAddressU || []
    let interiorNormalAddressV: (TextureAddressMode | undefined)[] = pobInteriorHit.meshPartNormalTextureAddressV || []
    let interiorPrimaryUvs: number[] = pobInteriorHit.meshPartPrimaryUvSetIndices || []
    let interiorSecondaryTex: (string | undefined)[] = pobInteriorHit.meshPartSecondaryTexturePaths || []
    let interiorSecondaryAddressU: (TextureAddressMode | undefined)[] = pobInteriorHit.meshPartSecondaryTextureAddressU || []
    let interiorSecondaryAddressV: (TextureAddressMode | undefined)[] = pobInteriorHit.meshPartSecondaryTextureAddressV || []
    let interiorSecondaryUvs: number[] = pobInteriorHit.meshPartSecondaryUvSetIndices || []
    let interiorShaderPaths: (string | undefined)[] = pobInteriorHit.meshPartShaderPaths || []
    let interiorChunkTrace: (string | undefined)[] = pobInteriorHit.meshPartChunkTrace || []
    let interiorEffectPaths: (string | undefined)[] = pobInteriorHit.meshPartEffectPaths || []

    const isLikelyExteriorTagged = (value: string): boolean => {
      const lower = value.toLowerCase()
      if (lower.includes('thed_exterior_')) return true
      if (lower.includes('thed_relief')) return true
      if (lower.includes('palace_window')) return true
      if (/(^|[^a-z])(exterior|facade|relief|statue|window)([^a-z]|$)/.test(lower)) return true
      return false
    }

    const keepInteriorPart = (index: number): boolean => {
      const tags = [
        interiorTextures[index],
        interiorNormals[index],
        interiorSecondaryTex[index],
        interiorShaderPaths[index],
        interiorChunkTrace[index],
      ].filter((v): v is string => Boolean(v && v.trim().length > 0))

      if (tags.length === 0) return true
      const joined = tags.join(' | ')
      return !isLikelyExteriorTagged(joined)
    }

    const keptIndices = interiorParts
      .map((_, index) => index)
      .filter((index) => keepInteriorPart(index))

    const pickOr = <T,>(source: T[], index: number, fallback: T): T => {
      if (index >= 0 && index < source.length) return source[index]
      return fallback
    }

    interiorParts = keptIndices.map((index) => interiorParts[index])
    interiorTextures = keptIndices.map((index) => pickOr(interiorTextures, index, ''))
    interiorTextureAddressU = keptIndices.map((index) => pickOr(interiorTextureAddressU, index, undefined))
    interiorTextureAddressV = keptIndices.map((index) => pickOr(interiorTextureAddressV, index, undefined))
    interiorTextureMipmapFilter = keptIndices.map((index) => pickOr(interiorTextureMipmapFilter, index, undefined))
    interiorTextureMinificationFilter = keptIndices.map((index) => pickOr(interiorTextureMinificationFilter, index, undefined))
    interiorTextureMagnificationFilter = keptIndices.map((index) => pickOr(interiorTextureMagnificationFilter, index, undefined))
    interiorNormals = keptIndices.map((index) => pickOr(interiorNormals, index, undefined))
    interiorNormalAddressU = keptIndices.map((index) => pickOr(interiorNormalAddressU, index, undefined))
    interiorNormalAddressV = keptIndices.map((index) => pickOr(interiorNormalAddressV, index, undefined))
    interiorPrimaryUvs = keptIndices.map((index) => pickOr(interiorPrimaryUvs, index, 0))
    interiorSecondaryTex = keptIndices.map((index) => pickOr(interiorSecondaryTex, index, undefined))
    interiorSecondaryAddressU = keptIndices.map((index) => pickOr(interiorSecondaryAddressU, index, undefined))
    interiorSecondaryAddressV = keptIndices.map((index) => pickOr(interiorSecondaryAddressV, index, undefined))
    interiorSecondaryUvs = keptIndices.map((index) => pickOr(interiorSecondaryUvs, index, 0))
    interiorShaderPaths = keptIndices.map((index) => pickOr(interiorShaderPaths, index, undefined))
    interiorChunkTrace = keptIndices.map((index) => pickOr(interiorChunkTrace, index, undefined))
    interiorEffectPaths = keptIndices.map((index) => pickOr(interiorEffectPaths, index, undefined))

    interiorPrimaryUvs = interiorParts.map((part, i) =>
      chooseInteriorPrimaryUvSet(part, interiorPrimaryUvs[i] ?? 0)
    )
    interiorParts = interiorParts.map((part) =>
      normalizeInteriorPartUvsForPreview(part)
    )
    interiorSecondaryTex = interiorSecondaryTex.map((path, i) => {
      const part = interiorParts[i]
      if (!part || !path) return undefined
      const vertexCount = Math.floor(part.positions.length / 3)
      const setIndex = interiorSecondaryUvs[i] ?? 0
      const hasUv0 = Boolean(part.uvs && part.uvs.length >= vertexCount * 2)
      const hasUv1 = Boolean(part.uvs1 && part.uvs1.length >= vertexCount * 2)
      const hasUvSet = Boolean(
        part.uvSets &&
        setIndex >= 0 &&
        setIndex < part.uvSets.length &&
        part.uvSets[setIndex]?.length >= vertexCount * 2,
      )
      if (setIndex === 1 && !hasUv1 && !hasUvSet) return undefined
      if (setIndex === 0 && !hasUv0 && !hasUvSet) return undefined
      if (setIndex > 1 && !hasUvSet) return undefined
      if (!hasUv0 && !hasUv1 && !hasUvSet) return undefined
      return path
    })
    interiorSecondaryAddressU = interiorSecondaryAddressU.map((mode, i) =>
      interiorSecondaryTex[i] ? mode : undefined
    )
    interiorSecondaryAddressV = interiorSecondaryAddressV.map((mode, i) =>
      interiorSecondaryTex[i] ? mode : undefined
    )
    interiorSecondaryUvs = interiorSecondaryUvs.map((setIndex, i) =>
      interiorSecondaryTex[i] ? setIndex : 0
    )

    finalMeshParts.push(pobInteriorHit.mesh)
    finalPathParts.push(pobInteriorHit.path)
    finalSourceParts.push(pobInteriorHit.sourceLabel)
    processMeshParts(
      interiorParts,
      interiorTextures,
      interiorTextureAddressU,
      interiorTextureAddressV,
      interiorTextureMipmapFilter,
      interiorTextureMinificationFilter,
      interiorTextureMagnificationFilter,
      interiorNormals,
      interiorNormalAddressU,
      interiorNormalAddressV,
      interiorPrimaryUvs,
      interiorSecondaryTex,
      interiorSecondaryAddressU,
      interiorSecondaryAddressV,
      interiorSecondaryUvs,
      interiorShaderPaths,
      interiorChunkTrace,
      interiorEffectPaths,
      'interior',
    )
  }

  // Deterministic fallback: if no authoritative/decoded exterior was resolved,
  // keep the declared appearance POB mesh as exterior even when interior cells exist.
  if (pobMeshFallback && !decodedMeshHit?.mesh) {
    finalMeshParts.push(pobMeshFallback)
    finalPathParts.push(appearancePath)
    finalSourceParts.push(appearanceSource.sourceLabel)
    processMeshParts([pobMeshFallback])
  }

  const finalMesh = finalMeshParts.length > 1
    ? combineMeshes(finalMeshParts)
    : finalMeshParts[0]

  const resolvedMeshPath = finalPathParts.length > 1
    ? `combined:${appearancePath}`
    : finalPathParts[0] ?? meshPath
  const resolvedSourceLabel = Array.from(new Set(finalSourceParts.filter(Boolean))).join('+') || appearanceSource.sourceLabel

  // Log final texture resolution for debugging
  console.log(`[Texture Resolution] Template: ${normalizedTemplate}`)
  console.log(`[Texture Resolution]   Appearance: ${appearancePath}`)
  console.log(`[Texture Resolution]   Shader: ${shaderPath || 'none'}`)
  console.log(`[Texture Resolution]   Mesh: ${resolvedMeshPath || 'none'}`)
  console.log(`[Texture Resolution]   \u2713 MAIN TEXTURE: ${texturePath || 'none'}`)
  if (normalTexturePath) console.log(`[Texture Resolution]   \u2713 NORMAL: ${normalTexturePath}`)
  if (renderMeshPartTexturePaths.length > 0) {
    console.log(`[Texture Resolution]   Mesh parts (${renderMeshPartTexturePaths.length}):`)
    renderMeshPartTexturePaths.forEach((path, i) => {
      console.log(`[Texture Resolution]     Part ${i}: ${path || 'none'}`)
    })
  }

  return {
    templatePath: normalizedTemplate,
    objectPath: normalizedTemplate,
    appearancePath,
    shaderPath,
    texturePath,
    textureAddressU,
    textureAddressV,
    textureMipmapFilter,
    textureMinificationFilter,
    textureMagnificationFilter,
    normalTexturePath,
    normalTextureAddressU,
    normalTextureAddressV,
    textureSourceLabel,
    normalTextureSourceLabel,
    effectPath,
    effectOptionCodes,
    vertexProgramPaths,
    pixelProgramPaths,
    shaderStageTexturePaths,
    shaderStageNormalTexturePaths,
    meshPath: resolvedMeshPath,
    sourceLabel: resolvedSourceLabel,
    mesh: finalMesh ?? undefined,
    meshParts: renderMeshParts.length ? renderMeshParts : undefined,
    meshPartTexturePaths: renderMeshPartTexturePaths.length
      ? renderMeshPartTexturePaths
      : undefined,
    meshPartTextureAddressU: renderMeshPartTextureAddressU.length
      ? renderMeshPartTextureAddressU
      : undefined,
    meshPartTextureAddressV: renderMeshPartTextureAddressV.length
      ? renderMeshPartTextureAddressV
      : undefined,
    meshPartTextureMipmapFilter: renderMeshPartTextureMipmapFilter.length
      ? renderMeshPartTextureMipmapFilter
      : undefined,
    meshPartTextureMinificationFilter: renderMeshPartTextureMinificationFilter.length
      ? renderMeshPartTextureMinificationFilter
      : undefined,
    meshPartTextureMagnificationFilter: renderMeshPartTextureMagnificationFilter.length
      ? renderMeshPartTextureMagnificationFilter
      : undefined,
    meshPartNormalTexturePaths: renderMeshPartNormalTexturePaths.length
      ? renderMeshPartNormalTexturePaths
      : undefined,
    meshPartNormalTextureAddressU: renderMeshPartNormalTextureAddressU.length
      ? renderMeshPartNormalTextureAddressU
      : undefined,
    meshPartNormalTextureAddressV: renderMeshPartNormalTextureAddressV.length
      ? renderMeshPartNormalTextureAddressV
      : undefined,
    meshPartPrimaryUvSetIndices: renderMeshPartPrimaryUvSetIndices.length
      ? renderMeshPartPrimaryUvSetIndices
      : undefined,
    meshPartSecondaryTexturePaths: renderMeshPartSecondaryTexturePaths.length
      ? renderMeshPartSecondaryTexturePaths
      : undefined,
    meshPartSecondaryTextureAddressU: renderMeshPartSecondaryTextureAddressU.length
      ? renderMeshPartSecondaryTextureAddressU
      : undefined,
    meshPartSecondaryTextureAddressV: renderMeshPartSecondaryTextureAddressV.length
      ? renderMeshPartSecondaryTextureAddressV
      : undefined,
    meshPartSecondaryUvSetIndices: renderMeshPartSecondaryUvSetIndices.length
      ? renderMeshPartSecondaryUvSetIndices
      : undefined,
    meshPartShaderPaths: renderMeshPartShaderPaths.length
      ? renderMeshPartShaderPaths
      : undefined,
    meshPartEffectPaths: renderMeshPartEffectPaths.length
      ? renderMeshPartEffectPaths
      : undefined,
    meshPartDomains: renderMeshPartDomains.length
      ? renderMeshPartDomains
      : undefined,
    meshPartChunkTrace: renderMeshPartChunkTrace.length
      ? renderMeshPartChunkTrace
      : undefined,
    hardpoints: collectedHardpoints.length ? collectedHardpoints : undefined,
    status: finalMesh ? 'mesh' : 'appearance-only',
  }
}






