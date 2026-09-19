import {test,expect} from '@playwright/test';
import {readFileSync,existsSync} from 'node:fs';
test('backflip assist reports applied forces on the replay clock',async({page})=>{
 test.skip(!existsSync('data/physics/backflip.json'));
 await page.route('**/api/physics/result',r=>r.fulfill({contentType:'application/json',body:readFileSync('data/physics/backflip.json','utf8')}));
 await page.goto('/?dataset=malecns');await page.getByRole('button',{name:'Load latest run'}).click();
 const panel=page.getByRole('region',{name:'Backflip assist observations'});await expect(panel).toContainText('assist off');
 await page.getByLabel('Activity time',{exact:true}).fill('1');await expect(panel).toContainText('assist active');
 await page.getByLabel('Activity time',{exact:true}).fill('2');await expect(panel).toContainText('assist off');
 await page.getByRole('button',{name:'Rewind activity'}).click();await expect(panel).toContainText('Sample at 0.00 s');
});
