// Run with Node and Playwright; reads the generated offline page, trains nothing.
const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const fs = require('node:fs');

(async () => {
  const options = { headless: true };
  if (process.env.CHROMIUM_EXECUTABLE) options.executablePath = process.env.CHROMIUM_EXECUTABLE;
  const browser = await chromium.launch(options);
  try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1080 } });
  const errors = [];
  page.on('pageerror', error => errors.push(String(error)));
  await page.goto(pathToFileURL(path.resolve('reports/forge/test-performance.html')).href);
  await page.locator('#matrix tbody tr').first().waitFor();
  assert.equal(await page.locator('#scope').inputValue(), 'selected');
  assert.equal(await page.locator('#stats .stat').count(), 4);
  await page.selectOption('#filter', 'zero');
  assert(await page.locator('#matrix tbody tr').count() > 0);
  assert(await page.locator('#matrix tbody tr:not(.zero)').count() === 0);
  await page.locator('#matrix tbody td button').first().click();
  assert(await page.locator('#detail').isVisible());
  assert((await page.locator('#detailBody').innerText()).includes('original'));
  await page.keyboard.press('Escape');
  assert(!await page.locator('#detail').isVisible());

  await page.selectOption('#scope', 'forge');
  await page.fill('#search', 'vector_anisotropic');
  await page.fill('#recipe', 'G-angle');
  await page.selectOption('#filter', 'all');
  assert(await page.locator('#matrix tbody tr').count() > 0);
  assert((await page.locator('#matrix tbody').innerText()).includes('FAIL'));
  await page.fill('#search', 'vector_unequal_width');
  assert((await page.locator('#matrix tbody').innerText()).includes('PASS'));
  const width = page.locator('#matrix tbody tr').filter({ hasText: 'Scoped diagnostic' });
  assert.equal(await width.count(), 2); // Seeds stay in two exact protocol rows.
  assert((await width.nth(0).innerText()).includes('PASS'));
  assert((await width.nth(1).innerText()).includes('PASS'));
  // Current unmeasured definitions do not inherit these historical passes.
  assert(await page.locator('#matrix tbody tr').filter({ hasText: 'Unmeasured declaration' }).count() > 0);

  await page.fill('#recipe', 'Atlas location');
  await page.fill('#search', 'two_pole');
  assert((await page.locator('#matrix tbody').innerText()).includes('PASS'));
  assert((await page.locator('#matrix tbody').innerText()).includes('FAIL'));
  assert((await page.locator('#matrix tbody').innerText()).includes('ERROR'));
  await page.selectOption('#filter', 'error');
  assert((await page.locator('#matrix tbody').innerText()).includes('ERROR'));
  assert(!(await page.locator('#matrix tbody').innerText()).includes('PASS'));

  await page.selectOption('#scope', 'selected');
  await page.fill('#recipe', '');
  await page.fill('#search', 'this-test-does-not-exist');
  await page.selectOption('#filter', 'all');
  assert(await page.locator('#empty').isVisible());
  assert((await page.locator('#rows').innerText()).includes('0 of 0'));
  await page.fill('#search', '');
  const directory = process.env.REPORT_SCREENSHOTS || '/tmp/forge-test-performance-browser';
  fs.mkdirSync(directory, { recursive: true });
  await page.screenshot({ path: path.join(directory, 'desktop.png'), fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({ path: path.join(directory, 'mobile.png'), fullPage: true });
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  assert.deepEqual(errors, []);
  console.log('PASS: offline render, filters, no-pass counts, G-angle seed separation, Atlas ERROR/FAIL/PASS, drill-down, keyboard close and mobile layout');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exit(1); });
