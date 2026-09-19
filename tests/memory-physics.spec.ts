import {test,expect} from '@playwright/test';
import {readFileSync,existsSync} from 'node:fs';
test('memory and locomotion remain separate activity sources on one playback clock',async({page})=>{
 test.skip(!existsSync('data/physics/memory-left.json'));
 await page.route('**/api/physics/result',r=>r.fulfill({contentType:'application/json',body:readFileSync('data/physics/memory-left.json','utf8')}));
 await page.goto('/?dataset=malecns');await page.getByRole('button',{name:'Load latest run'}).click();
 const panel=page.getByRole('region',{name:'Learned memory observations'});await expect(panel).toContainText('cue 1 / 0');
 await page.getByLabel('Physics activity model',{exact:true}).selectOption('memory');
 await page.getByLabel('Activity time',{exact:true}).fill('1');await expect(panel).toContainText('delay · cue 0 / 0');
 await page.getByLabel('Activity time',{exact:true}).fill('2');await expect(panel).toContainText('response · cue 0 / 0');
 await page.getByRole('button',{name:'Rewind activity'}).click();await expect(panel).toContainText('cue 1 / 0');
 await expect(page.getByLabel('Physics activity model',{exact:true})).toHaveValue('memory');
 await page.getByLabel('Physics activity model',{exact:true}).selectOption('locomotion');await expect(panel).toContainText('cue 1 / 0');
});
