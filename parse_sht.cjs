#!/usr/bin/env node
// Quick SHT IFF structure dumper
const fs = require('fs');

function parseChunks(buf, offset, end, depth) {
  const pad = '  '.repeat(depth);
  while (offset + 8 <= end) {
    const tag = buf.slice(offset, offset+4).toString('ascii');
    if (!/^[ -~]{4}$/.test(tag)) break;
    const len = buf.readUInt32BE(offset+4);
    if (len < 0 || offset + 8 + len > buf.length + 4) break;
    const isForm = tag === 'FORM' || tag === 'LIST' || tag === 'CAT ';
    if (isForm) {
      const type = buf.slice(offset+8, offset+12).toString('ascii');
      console.log(pad + '[' + offset + '] ' + tag + ':' + type + ' len=' + len);
      parseChunks(buf, offset+12, offset+8+len, depth+1);
    } else {
      let strs = [], s = '';
      const dataLen = Math.min(len, 512);
      for (let i = 0; i < dataLen; i++) {
        const b = buf[offset+8+i];
        if (b >= 32 && b <= 126) s += String.fromCharCode(b);
        else { if (s.length >= 2) strs.push(s); s = ''; }
      }
      if (s.length >= 2) strs.push(s);
      let hexData = '';
      if (len <= 8) hexData = ' hex=[' + Array.from(buf.slice(offset+8, offset+8+len)).map(b=>b.toString(16).padStart(2,'0')).join(' ') + ']';
      console.log(pad + '[' + offset + '] ' + tag + ' len=' + len + hexData + (strs.length ? ' | ' + strs.join(' | ') : ''));
    }
    offset += 8 + len;
    // align to 2 bytes
    if (offset % 2 !== 0) offset++;
  }
}

const files = process.argv.slice(2);
if (files.length === 0) {
  // default to naboo floor shader
  files.push('C:/Users/Hardy/Documents/naboo chain/shader/nboo_intr_palace_floor_ae9.sht');
  files.push('C:/Users/Hardy/Documents/naboo chain/shader/nboo_intr_throne_floor_ae9.sht');
  files.push('C:/Users/Hardy/Documents/naboo chain/shader/thed_marble_greylightslate_aes17.sht');
  files.push('C:/Users/Hardy/Documents/naboo chain/shader/thed_stair_adb13.sht');
}

for (const f of files) {
  console.log('\n=== ' + f + ' ===');
  try {
    const buf = fs.readFileSync(f);
    parseChunks(buf, 0, buf.length, 0);
  } catch(e) {
    console.log('ERROR:', e.message);
  }
}
