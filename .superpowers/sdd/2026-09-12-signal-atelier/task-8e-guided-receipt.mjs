/* global document */
import { chromium } from '@playwright/test';
import { mkdir, writeFile } from 'node:fs/promises';
const output='../../outputs/nn-forge-implementation/task-8e-round3';
await mkdir(output,{recursive:true});
const browser=await chromium.launch();const page=await browser.newPage({viewport:{width:1440,height:1024}});
await page.goto('http://127.0.0.1:5173');await page.getByRole('button',{name:'Run one training step'}).waitFor();
await page.getByRole('button',{name:'Lessons',exact:true}).click();await page.locator('.lesson-list').getByRole('button',{name:/Circle With One Hidden Layer/}).click();await page.getByRole('button',{name:'Start lesson and reset',exact:true}).click();
await page.getByRole('button',{name:'Exit lesson',exact:true}).waitFor();await page.getByRole('tab',{name:'Network',exact:true}).click();
const receipts=[];
for(const theme of ['light','dark']){await page.getByLabel('Color theme').selectOption(theme);await page.screenshot({path:`${output}/guided-${theme}.png`,fullPage:true});receipts.push({theme,...await page.evaluate(()=>({workspace:document.querySelector('.atelier-workspace').getBoundingClientRect().toJSON(),guide:document.querySelector('.atelier-lesson-host').getBoundingClientRect().toJSON()}))});}
await writeFile(`${output}/guided-geometry.json`,JSON.stringify(receipts,null,2));await browser.close();
