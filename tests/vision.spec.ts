import {test,expect} from '@playwright/test';
import {existsSync,readFileSync} from 'node:fs';
test('eye images follow the shared physics clock and rewind',async({page})=>{
 test.skip(!existsSync('data/physics/vision-current.json'));
 const body=readFileSync('data/physics/vision-current.json','utf8');
 await page.route('**/api/physics/result',r=>r.fulfill({contentType:'application/json',body}));
 await page.goto('/?dataset=malecns');
 await page.getByRole('button',{name:'Load latest run'}).click();
 const eye=page.getByRole('img',{name:'Left fly eye at 0.00 seconds',exact:true});
 await expect(eye).toBeVisible();const initial=await eye.getAttribute('src');
 await page.getByLabel('Activity time',{exact:true}).fill('1');
 const changed=page.getByRole('img',{name:'Left fly eye at 1.00 seconds',exact:true});
 await expect(changed).toBeVisible();await expect(changed).not.toHaveAttribute('src',initial!);
 await page.getByRole('button',{name:'Rewind activity'}).click();
 await expect(eye).toHaveAttribute('src',initial!);
});
