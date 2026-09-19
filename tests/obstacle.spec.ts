import {test,expect} from '@playwright/test';
import {readFileSync,existsSync} from 'node:fs';
test('obstacle sensor readings follow playback and rewind',async({page})=>{
 test.skip(!existsSync('data/physics/obstacle-training.json'));
 await page.route('**/api/physics/result',r=>r.fulfill({contentType:'application/json',body:readFileSync('data/physics/obstacle-training.json','utf8')}));
 await page.goto('/?dataset=malecns');await page.getByRole('button',{name:'Load latest run'}).click();
 const panel=page.getByRole('region',{name:'Obstacle observations'});await expect(panel).toBeVisible();const initial=await panel.textContent();
 await page.getByLabel('Activity time',{exact:true}).fill('1');await expect(panel).toContainText('Sample at 1.00 s');
 await page.getByRole('button',{name:'Rewind activity'}).click();await expect(panel).toHaveText(initial!);
 await page.getByLabel('Physical obstacles',{exact:true}).check();await expect(page.getByLabel('Obstacle layout',{exact:true})).toBeVisible();
});
