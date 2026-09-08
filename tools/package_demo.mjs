import { cp, mkdir, readdir, rm } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
const output=new URL('../demo/',import.meta.url);
// Only generated files inside the owned demo build directory are replaced.
await mkdir(output,{recursive:true});
for(const name of await readdir(output))await rm(new URL(name,output),{recursive:true,force:true});
await cp(fileURLToPath(new URL('../web/dist/',import.meta.url)),fileURLToPath(output),{recursive:true});
for(const folder of ['model','model-lite'])await rm(new URL(folder+'/g1.mjb',output),{force:true});
console.log('Packaged demo/ for GitHub Pages');
