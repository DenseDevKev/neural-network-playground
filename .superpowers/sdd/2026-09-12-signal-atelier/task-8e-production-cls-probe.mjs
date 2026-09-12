/* global window, document, PerformanceObserver, Node */
import { chromium, expect } from '@playwright/test';
import { writeFile } from 'node:fs/promises';
const browser=await chromium.launch();const receipts=[];
for(const width of [1440,768]){
 const page=await browser.newPage({viewport:{width,height:900}});await page.goto('http://127.0.0.1:4173');
 const transport=page.getByRole('region',{name:'Training controls',exact:true});await expect(transport).toHaveAttribute('data-model-generation',/.+/);
 await page.getByRole('button',{name:'Dismiss lesson suggestion'}).click();
 for(const name of ['Setup','Results','Inspect','Network'])await page.getByRole('tablist',{name:'Experiment workspace'}).getByRole('tab',{name,exact:true}).click();
 await page.evaluate(()=>document.fonts.ready);await page.getByRole('combobox',{name:'Steps per frame'}).selectOption('50');await page.getByRole('button',{name:'Start training',exact:true}).scrollIntoViewIfNeeded();
 await page.evaluate(()=>{
  window.__clsDetails=[];
  new PerformanceObserver(list=>{for(const entry of list.getEntries())if(!entry.hadRecentInput)window.__clsDetails.push({value:entry.value,time:entry.startTime,sources:entry.sources.map(source=>{
    const node=source.node;let textRect=null;if(node?.nodeType===Node.TEXT_NODE){const range=document.createRange();range.selectNodeContents(node);textRect=range.getBoundingClientRect().toJSON();}
    return {type:node?.nodeType,text:node?.textContent,parent:node?.parentElement?.outerHTML,parentRect:node?.parentElement?.getBoundingClientRect().toJSON(),textRect,previousRect:source.previousRect?.toJSON(),currentRect:source.currentRect?.toJSON()};
  })});}).observe({type:'layout-shift'});
 });
 await page.getByRole('button',{name:'Start training',exact:true}).click();await expect(transport).toHaveAttribute('data-status','running');await page.waitForTimeout(2500);
 receipts.push({width,step:await transport.getAttribute('data-model-step'),shifts:await page.evaluate(()=>window.__clsDetails)});
 await page.getByRole('button',{name:'Pause training',exact:true}).click();await page.close();
}
await browser.close();await writeFile('/tmp/task8e-production-cls-details.json',JSON.stringify(receipts,null,2));
