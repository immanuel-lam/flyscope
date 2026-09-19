// Pinned public data artifact; never substitute synthetic data on build failure.
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile, access } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
const manifest=JSON.parse(await readFile(new URL('../hosting-data.json',import.meta.url),'utf8'));
await mkdir('data/releases',{recursive:true});
const destination='data/releases/malecns-v1-viewer.tar.gz';
let bytes;
try { bytes=await readFile(destination); } catch {}
if(!bytes || createHash('sha256').update(bytes).digest('hex')!==manifest.sha256){
 const response=await fetch(manifest.url);
 if(!response.ok)throw new Error(`Dataset download failed: ${response.status}`);
 bytes=Buffer.from(await response.arrayBuffer());
 if(createHash('sha256').update(bytes).digest('hex')!==manifest.sha256)throw new Error('Dataset checksum mismatch');
 await writeFile(destination,bytes);
}
await mkdir('public',{recursive:true});
execFileSync('tar',['-xzf',destination,'-C','public']);
for(const name of ['catalog.json','graph.json','outgoing-counts.bin',...Array.from({length:128},(_,i)=>`connections/${i}.bin`)])await access(`public/malecns/${name}`);
console.log('Verified and extracted MaleCNS v1 viewer assets.');
