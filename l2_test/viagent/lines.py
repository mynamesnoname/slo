"""Rest-frame spectral line reference table (from kb/lines.md)."""

# name, rest wavelength (A), type (em/abs), width class
LINE_TABLE = [
    {"name": "Lyα", "rest": 1215.7, "type": "em", "width_class": "broad"},
    {"name": "C IV", "rest": 1549.0, "type": "em", "width_class": "broad"},
    {"name": "He II", "rest": 1640.4, "type": "em", "width_class": "both"},
    {"name": "C III]", "rest": 1909.0, "type": "em", "width_class": "broad"},
    {"name": "Mg II", "rest": 2800.0, "type": "em", "width_class": "broad"},
    {"name": "Mg II_abs", "rest": 2800.0, "type": "abs", "width_class": "absorption"},
    {"name": "[Ne V]", "rest": 3426.0, "type": "em", "width_class": "narrow"},
    {"name": "[O II]", "rest": 3727.0, "type": "em", "width_class": "narrow"},
    {"name": "Ca K_abs", "rest": 3934.8, "type": "abs", "width_class": "absorption"},
    {"name": "Ca H_abs", "rest": 3969.6, "type": "abs", "width_class": "absorption"},
    {"name": "Hε", "rest": 3970.1, "type": "em", "width_class": "both"},
    {"name": "Hε_abs", "rest": 3970.1, "type": "abs", "width_class": "absorption"},
    {"name": "Hδ", "rest": 4102.9, "type": "em", "width_class": "both"},
    {"name": "Hδ_abs", "rest": 4102.9, "type": "abs", "width_class": "absorption"},
    {"name": "G-band_abs", "rest": 4305.6, "type": "abs", "width_class": "absorption"},
    {"name": "Hγ", "rest": 4341.7, "type": "em", "width_class": "both"},
    {"name": "Hγ_abs", "rest": 4341.7, "type": "abs", "width_class": "absorption"},
    {"name": "Hβ", "rest": 4862.7, "type": "em", "width_class": "both"},
    {"name": "Hβ_abs", "rest": 4862.7, "type": "abs", "width_class": "absorption"},
    {"name": "[O III]a", "rest": 4960.3, "type": "em", "width_class": "narrow"},
    {"name": "[O III]b", "rest": 5008.2, "type": "em", "width_class": "narrow"},
    {"name": "Mg I_abs", "rest": 5176.7, "type": "abs", "width_class": "absorption"},
    {"name": "Na D_abs", "rest": 5895.6, "type": "abs", "width_class": "absorption"},
    {"name": "[N II]a", "rest": 6549.8, "type": "em", "width_class": "narrow"},
    {"name": "Hα", "rest": 6564.6, "type": "em", "width_class": "both"},
    {"name": "Hα_abs", "rest": 6564.6, "type": "abs", "width_class": "absorption"},
    {"name": "[N II]b", "rest": 6585.3, "type": "em", "width_class": "narrow"},
    {"name": "[S II]a", "rest": 6718.3, "type": "em", "width_class": "narrow"},
    {"name": "[S II]b", "rest": 6732.7, "type": "em", "width_class": "narrow"},
    {"name": "CaT1_abs", "rest": 8498.0, "type": "abs", "width_class": "absorption"},
    {"name": "CaT2_abs", "rest": 8542.0, "type": "abs", "width_class": "absorption"},
    {"name": "CaT3_abs", "rest": 8662.0, "type": "abs", "width_class": "absorption"},
]

# Rest-frame doublet spacings (A) - from kb/lines.md
DOUBLETS = {
    "[O III]": {"rest": 5008.2, "sep": 47.9, "ratio": 3.0, "partner": 4960.3},
    "[S II]": {"rest": 6732.7, "sep": 14.4, "ratio": 1.0, "partner": 6718.3},
    "[N II]": {"rest": 6585.3, "sep": 35.5, "ratio": 3.0, "partner": 6549.8},
    "Ca K/H": {"rest": 3969.6, "sep": 34.8, "ratio": None, "partner": 3934.8},
}
