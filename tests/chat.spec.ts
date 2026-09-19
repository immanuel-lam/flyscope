import {test,expect} from '@playwright/test';
import {existsSync} from 'node:fs';
test('MaleCNS opens by default with working FlyGPT side chat and actual neural state',async({page})=>{
 test.skip(!existsSync('models/malecns-chat/runtime.npz')||!existsSync('public/malecns/catalog.json'),'Prepare data and train the chat model first');
 test.setTimeout(60000);const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('/');
 await expect(page.getByLabel('Dataset',{exact:true})).toHaveValue('full');
 await expect(page.getByRole('tab',{name:'FlyGPT',exact:true})).toHaveAttribute('aria-selected','true');
 await page.getByLabel('Message',{exact:true}).fill('Hi');
 await page.getByRole('button',{name:'Send',exact:true}).click();
 await expect(page.locator('.chat-message.assistant')).toContainText('Hello',{timeout:20000});
 await expect(page.getByTestId('chat-provenance')).toContainText('real graph enabled');
 await expect(page.locator('.brain-viewport canvas')).toHaveAttribute('data-recorded-cells','512');
 await page.getByRole('tab',{name:'Neuron inspector',exact:true}).click();
 await expect(page.getByLabel('Find neurons')).toBeVisible();
 await page.getByRole('tab',{name:'FlyGPT',exact:true}).click();
 await expect(page.locator('.chat-message.assistant')).toContainText('Hello');
 expect(errors).toEqual([]);
});
