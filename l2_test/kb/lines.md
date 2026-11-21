# Spectral Line Reference

## Related Knowledge

- Classification diagnostics (ELG, LRG, QSO, Star rules, fatal problems): see `kb/classification.md`
- Ionization physics (redshift anchoring priorities, consistency rules, outflow): see `kb/ionization.md`
- Emission-absorption composite profiles (Mg II, Hα, Hβ): see `kb/composite_profile.md`

## Rest-Frame Line Table

| Name       | λ_rest (Å) | Type | Width Class |
| ---------- | ------------ | ---- | ----------- |
| Lyα       | 1216.0       | em   | broad       |
| C IV       | 1549.0       | em   | broad       |
| He II      | 1640.4       | em   | both        |
| C III]     | 1909.0       | em   | broad       |
| Mg II      | 2800.0       | em   | broad       |
| Mg II_abs  | 2800.0       | abs  | absorption  |
| [Ne V]     | 3426.0       | em   | narrow      |
| [O II]     | 3727.0       | em   | narrow      |
| Ca K_abs   | 3934.8       | abs  | absorption  |
| Ca H_abs   | 3969.6       | abs  | absorption  |
| Hε        | 3970.1       | em   | both        |
| Hε_abs    | 3970.1       | abs  | absorption  |
| Hδ        | 4102.9       | em   | both        |
| Hδ_abs    | 4102.9       | abs  | absorption  |
| G-band_abs | 4305.6       | abs  | absorption  |
| Hγ        | 4341.7       | em   | both        |
| Hγ_abs    | 4341.7       | abs  | absorption  |
| Hβ        | 4862.7       | em   | both        |
| Hβ_abs    | 4862.7       | abs  | absorption  |
| [O III]a   | 4960.3       | em   | narrow      |
| [O III]b   | 5008.2       | em   | narrow      |
| Mg I_abs   | 5176.7       | abs  | absorption  |
| Na D_abs   | 5895.6       | abs  | absorption  |
| [N II]a    | 6549.8       | em   | narrow      |
| Hα        | 6564.6       | em   | both        |
| Hα_abs    | 6564.6       | abs  | absorption  |
| [N II]b    | 6585.3       | em   | narrow      |
| [S II]a    | 6718.3       | em   | narrow      |
| [S II]b    | 6732.7       | em   | narrow      |
| CaT1_abs   | 8498.0       | abs  | absorption  |
| CaT2_abs   | 8542.0       | abs  | absorption  |
| CaT3_abs   | 8662.0       | abs  | absorption  |

## Doublet Rules

**General principle — spacing alone does NOT confirm a doublet.** The real diagnostic question is whether **both components are physically real features**. Specifically:

- Is the weaker component a genuine peak/trough, or just noise at roughly the right position?
- In the OH airglow zone (>7800 Å), if the weaker line is unusually bright, suspect OH skyline contamination masquerading as the doublet partner. Flag explicitly.

### [O III] (4960.3 / 5008.2 Å)

- Rest separation: 47.9 Å. Observed separation: 47.9 × (1+z) Å.
- Spacing check: |λ_obs(b) − λ_obs(a) − 47.9×(1+z)| < 5 Å. Wrong spacing → not [O III].
- Amplitude ratio: b : a ≈ 3:1. Reversed ratio (a > b) disqualifies.

### [S II] (6718.3 / 6732.7 Å)

- Rest separation: 14.4 Å. Observed: 14.4 × (1+z) Å.
- Amplitude ratio: a ≈ b (both similar strength).

### [N II] (6549.8 / 6585.3 Å)

- Rest separation: 35.5 Å. Observed: 35.5 × (1+z) Å.
- Amplitude ratio: a : b ≈ 1:3 (b brighter).

### Ca K/H absorption (3934.8 / 3969.6 Å)

- Rest separation: 34.8 Å. Observed: 34.8 × (1+z) Å.
- Ca K MUST be deeper than Ca H. A single absorption without its partner is likely a misidentification.

### [O II] Unresolved Doublet (3726.0 / 3729.0 Å)

The [O II] 3727 line is a close doublet (rest separation 2.8–3.0 Å). At DESI resolution, the doublet may be **unresolved** — sometimes it has two split spikes, but it may also appear as a single blended emission peak with certain width. However, it should not be a one-or-two pixel narrow peak, that may be some kind of artifact.

### Lyα Forest and DLA

Lyα at z ≳ 1.5 may be accompanied by the **Lyα forest** — a dense series of narrow H I absorption lines blueward of the Lyα emission peak. A **DLA** (Damped Lyman-α Absorber) produces a broad, saturated absorption trough immediately blueward of Lyα.

**Asymmetric rule — confirmatory only**:

- **Forest visible**: A series of narrow absorption lines blueward of Lyα, with flux systematically depressed blueward compared to redward. This is **strong positive confirmation** of the Lyα identification and redshift.
- **Forest NOT visible but observable** (λ_pred − 200 Å > 4000 Å): Flag as "Lyα identification uncertain." Downgrade confidence but do NOT exclude the hypothesis.
- **Forest beyond blue edge** (λ_pred − 200 Å < 4000 Å): **Zero information.** Do NOT use as evidence against the hypothesis.
- **DLA check**: If Lyα appears narrower/weaker than expected, a DLA trough may be absorbing the blue wing — this explains the width anomaly and should NOT be held against the hypothesis.

## Blue Edge Noise Zone (λ_obs < 4000 Å for DESI)

The DESI blue arm (3600–4000 Å) may have high noise. Spectral features here require additional scrutiny. Treat them as **presumptively unreliable until visually confirmed**.

## OH Airglow Zone and Skyline Contamination

### OH Airglow (Red edge: > 7000 Å, strongest > 9000 Å)

Hydroxyl (OH) molecular bands produce dense, bright narrow emission lines. Even after sky subtraction, residual OH lines appear at fixed observed wavelengths.

- **Evaluation**: Any line with λ_pred > 7800 Å should be flagged as "OH zone." Features indistinguishable from the OH residual forest should be treated as unreliable and must not be used as sole evidence for a line identification — OH skylines can masquerade as astrophysical lines. Visually dominant peaks that do NOT match any known skyline position (see table below) may be real but should still carry an OH contamination caveat. Read the spectrum ±150 Å before accepting any OH-zone line.
- A match within ±10 Å of a known bright skyline is strong evidence of atmospheric origin — treat the astrophysical line as undetected (the feature is atmospheric, not astrophysical).

### OI Airglow (Visible: 5577, 6300, 6364 Å)

Atomic oxygen forbidden transitions at fixed observed wavelengths:

| λ_obs (Å) | Transition      | Notes                                                             |
| ----------- | --------------- | ----------------------------------------------------------------- |
| 5577.3      | [OI] green line | Strong, isolated. Often the brightest skyline in the visible band |
| 6300.3      | [OI] red line   | Weaker, but narrow and persistent                                 |
| 6363.8      | [OI] red line   | Companion to 6300.3, ratio 6300/6364 ≈ 3:1                       |

Unlike OH which is a red-edge problem, OI lines can contaminate features ANYWHERE in the visible spectrum (4000–7000 Å). Any narrow emission feature near these wavelengths should be checked against OI contamination regardless of the claimed redshift.

### Comprehensive Skyline / Atmospheric Feature Table

| Type                   | Name            | Wavelength (Å) | Notes                                                                  |
| ---------------------- | --------------- | --------------- | ---------------------------------------------------------------------- |
| Airglow Emission       | [O I]           | 5577.3          | Most famous skyline; easily produces spurious narrow emission peaks    |
| Airglow Emission       | Na D            | 5890, 5896      | Upper-atmosphere sodium layer emission doublet                         |
| Airglow Emission       | [O I]           | 6300.3          | Often confused with astrophysical [O I] 6300 in AGN                    |
| Airglow Emission       | [O I]           | 6363.8          | 3:1 doublet companion to 6300.3                                        |
| Atmospheric Absorption | O₂ B-band      | 6867            | Strong absorption band; affects continuum and weak-line identification |
| OH Airglow Forest      | OH Meinel Bands | 7000–7400      | Dense OH emission lines begin                                          |
| Atmospheric Absorption | O₂ A-band      | 7605            | One of the strongest atmospheric absorption bands                      |
| OH Airglow Forest      | OH Meinel Bands | 7800–9800      | Extremely dense OH forest                                              |

When evaluating ANY narrow emission feature, cross-reference λ_obs against this table. 

Lines that may fall in the OH zone at higher redshift include: [O II] (3727), Ca K/H (3935/3970), Hδ (4103) — rest-frame optical features that are key diagnostics for galaxy classification. When these fall beyond 7800 Å, their reliability is compromised and this must be noted in the synthesis verdict.
