/* global console, document */
import { webkit, chromium } from '@playwright/test';
for (const [name, engine] of [['webkit', webkit], ['chromium', chromium]]) {
 const browser=await engine.launch();const page=await browser.newPage({viewport:{width:320,height:844}});await page.goto('http://127.0.0.1:5173');
 const nodes=page.getByRole('group',{name:'Select a neuron'}).getByRole('button');await nodes.first().waitFor();
 const focus=()=>page.evaluate(()=>({name:document.activeElement?.getAttribute('aria-label'),tag:document.activeElement?.tagName,text:document.activeElement?.textContent?.slice(0,80)}));
 for(const key of ['Tab','Alt+Tab']){await nodes.first().focus();await page.keyboard.press('Enter');console.log(name,key,'after Enter',await focus());await page.keyboard.press(key);console.log(name,key,'after traversal',await focus());}
 await browser.close();
}
