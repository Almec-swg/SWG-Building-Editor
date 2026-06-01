import fs from 'fs';
import pako from 'pako';

// Parse TRE file
function parseTre(buffer) {
  const view = new DataView(buffer);
  const magic = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
  
  if (magic !== 'EERT' && magic !== '5EERT') {
    throw new Error('Not a valid TRE file');
  }
  
  const recordCount = view.getUint32(36, true);
  const nameTableOffset = view.getUint32(44, true);
  const nameTableCompressedSize = view.getUint32(48, true);
  const nameTableUncompressedSize = view.getUint32(52, true);
  
  // Decompress name table
  const nameTableCompressed = new Uint8Array(buffer, nameTableOffset, nameTableCompressedSize);
  const nameTableBytes = pako.inflate(nameTableCompressed);
  const nameTableDecoder = new TextDecoder('utf-8');
  const nameTableText = nameTableDecoder.decode(nameTableBytes);
  const names = nameTableText.split('\0').filter(n => n.length > 0);
  
  // Parse records
  const records = [];
  let offset = 148;
  
  for (let i = 0; i < recordCount; i++) {
    const checksum = view.getUint32(offset, true);
    const dataOffset = view.getUint32(offset + 4, true);
    const compressedSize = view.getUint32(offset + 8, true);
    const uncompressedSize = view.getUint32(offset + 12, true);
    const compressionType = view.getUint8(offset + 16);
    const nameOffset = view.getUint32(offset + 20, true);
    
    records.push({
      checksum,
      dataOffset,
      compressedSize,
      uncompressedSize,
      compressionType,
      name: names[i] || `unknown_${i}`
    });
    
    offset += 24;
  }
  
  return { records, buffer };
}

// Extract file from TRE
function extractFile(tre, filename) {
  const normalized = filename.toLowerCase().replace(/\\/g, '/');
  const record = tre.records.find(r => r.name.toLowerCase() === normalized);
  
  if (!record) {
    return null;
  }
  
  const data = new Uint8Array(tre.buffer, record.dataOffset, record.compressedSize);
  
  if (record.compressionType === 2) {
    return pako.inflate(data).buffer;
  }
  
  return data.buffer;
}

// Parse IFF
function parseIff(buffer) {
  const view = new DataView(buffer);
  const bytes = new Uint8Array(buffer);
  
  function readTag(offset) {
    return String.fromCharCode(bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]);
  }
  
  function parseChunk(offset, end, depth = 0) {
    if (offset + 8 > end) return null;
    
    const tag = readTag(offset);
    const size = view.getUint32(offset + 4, true);
    const dataStart = offset + 8;
    const dataEnd = dataStart + size;
    
    if (dataEnd > end) return null;
    
    const indent = '  '.repeat(depth);
    const isContainer = tag === 'FORM' || tag === 'LIST' || tag === 'CAT ';
    
    let result = [];
    
    if (isContainer) {
      const type = readTag(dataStart);
      result.push(`${indent}${tag}:${type} size=${size}`);
      
      let childOffset = dataStart + 4;
      while (childOffset + 8 <= dataEnd) {
        const childTag = readTag(childOffset);
        const childSize = view.getUint32(childOffset + 4, true);
        
        if (childOffset + 8 + childSize > dataEnd) break;
        
        const childResult = parseChunk(childOffset, dataEnd, depth + 1);
        if (childResult) result.push(...childResult);
        
        childOffset += 8 + childSize;
        if (childOffset % 2 !== 0) childOffset++;
      }
    } else {
      // Data chunk - show hex
      const dataLen = Math.min(size, 32);
      const hexData = [];
      for (let i = 0; i < dataLen; i++) {
        hexData.push(bytes[dataStart + i].toString(16).padStart(2, '0'));
      }
      const hex = hexData.join(' ');
      const more = size > 32 ? '...' : '';
      result.push(`${indent}${tag} size=${size} hex=[${hex}${more}]`);
    }
    
    return result;
  }
  
  return parseChunk(0, buffer.byteLength, 0);
}

// Main
async function main() {
  const treFiles = [
    'C:/Users/Hardy/Documents/naboo chain/mtg_patch_004_appearance_04.tre',
    'C:/Users/Hardy/Documents/naboo chain/mtg_patch_001_appearance_01.tre',
    'C:/Users/Hardy/Documents/naboo chain/appearance_01.tre'
  ];
  
  console.log('=== Analyzing shader: shader/thed_palace_window_asb13.sht ===\n');
  
  for (const treFile of treFiles) {
    if (!fs.existsSync(treFile)) continue;
    
    console.log(`Checking TRE: ${treFile}`);
    const treBuffer = fs.readFileSync(treFile).buffer;
    const tre = parseTre(treBuffer);
    
    const shaderData = extractFile(tre, 'shader/thed_palace_window_asb13.sht');
    if (shaderData) {
      console.log(`Found shader! Size: ${shaderData.byteLength} bytes\n`);
      console.log('IFF Structure:\n');
      const structure = parseIff(shaderData);
      structure.forEach(line => console.log(line));
      process.exit(0);
    }
  }
  
  console.log('Shader not found in any TRE file');
  process.exit(1);
}

main().catch(err => {
  console.error('Error:', err);
  process.exit(1);
});
