import { inflate } from 'pako'

export type TreCompression = 0 | 2

export interface TreHeader {
  records: number
  recordStart: number
  recordCompression: TreCompression
  recordCompressed: number
  nameCompression: TreCompression
  nameCompressed: number
  nameUncompressed: number
}

export interface TreRecordMeta {
  checksum: number
  dataUncompressed: number
  dataOffset: number
  dataCompression: TreCompression
  dataCompressed: number
  nameOffset: number
  name: string
}

export interface ParsedTreArchive {
  header: TreHeader
  records: TreRecordMeta[]
}

function readU32(view: DataView, offset: number): number {
  return view.getUint32(offset, true)
}

function decodeBlock(data: Uint8Array, compression: TreCompression): Uint8Array {
  if (compression === 0) return data
  if (compression === 2) return inflate(data)
  throw new Error(`Unsupported TRE compression method: ${compression}`)
}

function readCString(bytes: Uint8Array, start: number): string {
  let end = start
  while (end < bytes.length && bytes[end] !== 0) end += 1
  return new TextDecoder('latin1').decode(bytes.subarray(start, end))
}

export async function parseTreArchive(file: File): Promise<ParsedTreArchive> {
  const buffer = await file.arrayBuffer()
  const view = new DataView(buffer)
  if (buffer.byteLength < 36) throw new Error('TRE file too small')

  const magic = String.fromCharCode(
    view.getUint8(0),
    view.getUint8(1),
    view.getUint8(2),
    view.getUint8(3),
    view.getUint8(4),
    view.getUint8(5),
    view.getUint8(6),
    view.getUint8(7),
  )

  // TRE/TREE v0005 little endian magic in bytes is usually "EERT5000".
  if (magic !== 'EERT5000' && magic !== 'TREE0005') {
    throw new Error(`Invalid TRE header magic: ${magic}`)
  }

  const header: TreHeader = {
    records: readU32(view, 8),
    recordStart: readU32(view, 12),
    recordCompression: readU32(view, 16) as TreCompression,
    recordCompressed: readU32(view, 20),
    nameCompression: readU32(view, 24) as TreCompression,
    nameCompressed: readU32(view, 28),
    nameUncompressed: readU32(view, 32),
  }

  const recordBlockStart = header.recordStart
  const recordBlockEnd = recordBlockStart + header.recordCompressed
  const nameBlockStart = recordBlockEnd
  const nameBlockEnd = nameBlockStart + header.nameCompressed

  if (
    recordBlockStart < 36 ||
    recordBlockEnd > buffer.byteLength ||
    nameBlockEnd > buffer.byteLength
  ) {
    throw new Error('TRE metadata block bounds are invalid')
  }

  const recordBlockRaw = new Uint8Array(buffer, recordBlockStart, header.recordCompressed)
  const nameBlockRaw = new Uint8Array(buffer, nameBlockStart, header.nameCompressed)

  const recordBlock = decodeBlock(recordBlockRaw, header.recordCompression)
  const nameBlock = decodeBlock(nameBlockRaw, header.nameCompression)

  const recordSize = 24
  if (recordBlock.length < header.records * recordSize) {
    throw new Error('TRE record block is shorter than expected')
  }

  const records: TreRecordMeta[] = []
  const recordView = new DataView(
    recordBlock.buffer,
    recordBlock.byteOffset,
    recordBlock.byteLength,
  )

  for (let i = 0; i < header.records; i += 1) {
    const base = i * recordSize
    const meta: Omit<TreRecordMeta, 'name'> = {
      checksum: readU32(recordView, base + 0),
      dataUncompressed: readU32(recordView, base + 4),
      dataOffset: readU32(recordView, base + 8),
      dataCompression: readU32(recordView, base + 12) as TreCompression,
      dataCompressed: readU32(recordView, base + 16),
      nameOffset: readU32(recordView, base + 20),
    }

    if (meta.nameOffset >= nameBlock.length) {
      throw new Error('TRE record name offset is out of bounds')
    }

    const name = readCString(nameBlock, meta.nameOffset).replace(/\\/g, '/')
    records.push({ ...meta, name })
  }

  return { header, records }
}

export async function extractTreRecord(file: File, record: TreRecordMeta): Promise<File> {
  const buffer = await file.arrayBuffer()
  const start = record.dataOffset
  const end = start + record.dataCompressed

  if (start < 0 || end > buffer.byteLength || start >= end) {
    throw new Error('TRE record data bounds are invalid')
  }

  const raw = new Uint8Array(buffer, start, record.dataCompressed)
  const decoded = decodeBlock(raw, record.dataCompression)
  const payload = Uint8Array.from(decoded).buffer

  return new File([payload], record.name, { type: 'application/octet-stream' })
}
