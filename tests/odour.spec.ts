import {test,expect} from '@playwright/test';
import {readFileSync,existsSync} from 'node:fs';
test('antenna observations follow physics playback and rewind',async({page})=>{
 test.skip(!existsSync('data/physics/odour-left.json'));
 await page.route('**/api/physics/result',r=>r.fulfill({contentType:'application/json',body:readFileSync('data/physics/odour-left.json','utf8')}));
 await page.goto('/?dataset=malecns');await page.getByRole('button',{name:'Load latest run'}).click();
 const panel=page.getByRole('region',{name:'Fly odour observations'});
 await expect(panel).toBeVisible();const initial=await panel.textContent();
 await page.getByLabel('Activity time',{exact:true}).fill('1');await expect(panel).toContainText('Sample at 1.00 s');expect(await panel.textContent()).not.toBe(initial);
 await page.getByRole('button',{name:'Rewind activity'}).click();await expect(panel).toHaveText(initial!);
});
