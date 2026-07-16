import { expect, test } from '@playwright/test'

test.beforeEach(async ({ page }) => {
  await page.goto('/')
  await expect(page.getByRole('heading', { name: /group of people riding skis/i })).toBeVisible()
  await expect(page.locator('.image-stage > img').first()).toHaveJSProperty('complete', true)
})

test('lab exposes the exact trace and clipping overlay', async ({ page }, testInfo) => {
  const isMobile = testInfo.project.name.includes('mobile')
  const nativeOutputs = page.locator('.native-output:visible')
  await expect(page.getByRole('heading', { name: /futuristic glass research station/i })).toBeVisible()
  await expect(nativeOutputs).toHaveCount(isMobile ? 1 : 3)
  await expect(page.locator('.native-image-stage img').first()).toHaveJSProperty('naturalWidth', 1024)

  if (isMobile) {
    await page.locator('.native-hero-switch').getByRole('button', { name: 'CFG' }).click()
    await expect(page.locator('.native-output[data-scheme="cfg"]')).toBeVisible()
  }

  const visiblePanels = page.locator('.comparison-panel:visible')
  const visibleExperimentImages = page.locator('.image-stage > img:first-child:visible')
  await expect(visiblePanels).toHaveCount(isMobile ? 1 : 3)
  await expect(visibleExperimentImages).toHaveCount(isMobile ? 1 : 3)
  await expect(page.getByText('Step 12/12')).toBeVisible()
  expect(await visibleExperimentImages.evaluateAll((images) => images.map((image) => image.getAttribute('data-render-mode'))))
    .toEqual(Array(isMobile ? 1 : 3).fill('retina'))

  await page.getByRole('button', { name: 'Raw 512' }).click()
  expect(await visibleExperimentImages.evaluateAll((images) => images.map((image) => image.getAttribute('data-render-mode'))))
    .toEqual(Array(isMobile ? 1 : 3).fill('raw'))
  await page.getByRole('button', { name: 'Retina' }).click()

  await page.getByRole('button', { name: 'X-ray' }).click()
  await expect(page.locator('.mask-layer:visible')).toHaveCount(isMobile ? 1 : 3)

  await page.getByTitle('Replay trace').click()
  await expect(page.getByText('Step 1/12')).toBeVisible({ timeout: 2500 })
  await page.getByTitle('Pause trace').click()

  if (isMobile) {
    await page.locator('.mobile-scheme-switch').getByRole('button', { name: 'CFG' }).click()
    await expect(page.locator('.comparison-panel[data-scheme="cfg"]')).toBeVisible()
  }

  await page.getByRole('button', { name: 'X-ray' }).click()
  const slider = page.getByRole('slider', { name: 'Denoising step' })
  await slider.focus()
  await slider.press('End')
  await expect(page.getByText('Step 12/12')).toBeVisible()
  if (isMobile) {
    await page.locator('.mobile-scheme-switch').getByRole('button', { name: 'Fitted' }).click()
    await expect(page.locator('.comparison-panel[data-scheme="fitted"]')).toBeVisible()
  }
  await page.evaluate(() => window.scrollTo(0, 0))

  await expect(page).toHaveScreenshot('lab.png', { fullPage: true })
})

test('atlas cells drive the evidence table', async ({ page }, testInfo) => {
  const isMobile = testInfo.project.name.includes('mobile')
  await page.getByRole('button', { name: 'Atlas' }).click()
  await expect(page.getByRole('heading', { name: 'Where the one-line repair matters.' })).toBeVisible()
  await page.getByRole('button', { name: /w 8, N 32/ }).click()
  await expect(page.getByRole('heading', { name: 'w = 8, N = 32' })).toBeVisible()
  await expect(page.getByRole('cell', { name: '18.67' })).toBeVisible()
  await expect(page).toHaveScreenshot(`atlas-${isMobile ? 'mobile' : 'desktop'}.png`, { fullPage: true })
})

test('build view carries the sampler and provenance contract', async ({ page }, testInfo) => {
  const isMobile = testInfo.project.name.includes('mobile')
  await page.getByRole('button', { name: 'Build' }).click()
  await expect(page.getByRole('heading', { name: 'From theorem to inspectable software.' })).toBeVisible()
  await expect(page.getByText('r ** (1.0 + w) - r')).toBeVisible()
  await expect(page.getByText('d954e5686324ac71c6867afdf68b94d0db44c0cb1fc0642f92f1a3c284fead4a')).toBeVisible()
  await expect(page).toHaveScreenshot(`build-${isMobile ? 'mobile' : 'desktop'}.png`, { fullPage: true })
})
