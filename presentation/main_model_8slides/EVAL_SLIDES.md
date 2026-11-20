Run `python presentation/main_model_8slides/generate_eval_slides.py` from the repository root. This standard-library script copies reported values; it does not rescore predictions.

Tier 1 table and plot use tier1_summary.csv. Tier 2 table uses tier2_summary.csv. Tier 0 uses tier0_status.csv. Tier 3 uses verdicts.csv and frozen thresholds stated in REPORT_TIER3_VI.md.

{
  "tier0_source": "report/tier0_status.csv",
  "tier1_source": "report/tier1_summary.csv",
  "tier1_display": [
    [
      "Hron",
      "0.0897",
      "0.0906",
      "0.1147"
    ],
    [
      "Tsagris",
      "0.1009",
      "0.0993",
      "0.1219"
    ],
    [
      "MICoDA A18",
      "0.1043",
      "0.1069",
      "0.1185"
    ],
    [
      "Same-day mean",
      "0.0311",
      "0.0304",
      "0.0364"
    ],
    [
      "Main",
      "0.0471",
      "0.0237",
      "0.0581"
    ],
    [
      "No metadata",
      "0.0516",
      "0.0233",
      "0.0644"
    ]
  ],
  "tier2_source": "report/tier2_summary.csv",
  "tier2_display": [
    [
      "Hron",
      "+0.054",
      "0.0485",
      "102",
      "6.947"
    ],
    [
      "Tsagris",
      "-0.052",
      "0.0646",
      "589",
      "18.441"
    ],
    [
      "MICoDA A18",
      "-0.055",
      "0.0652",
      "1259",
      "24.028"
    ],
    [
      "Same-day mean",
      "+0.066",
      "0.0565",
      "83",
      "0.362"
    ],
    [
      "Main",
      "-0.005",
      "0.0591",
      "433",
      "72.654"
    ],
    [
      "No metadata",
      "-0.062",
      "0.0598",
      "793",
      "106.732"
    ]
  ],
  "tier3_source": "report_tier3/verdicts.csv",
  "plot_source": "report/tier1_summary.csv: mae_prop_mean, mae_prop_sd",
  "sign_source": "report/tier1_pairs.csv: main vs main_nometa, mae_prop, sign_p",
  "plot_exact_values": [
    {
      "x": "0.78",
      "mean": "0.08970919762800285",
      "sd": "0.012979968054667116",
      "mask": "whole_feature",
      "model": "hron"
    },
    {
      "x": "1.0",
      "mean": "0.09061274016221607",
      "sd": "0.0033821420587300754",
      "mask": "random_cell",
      "model": "hron"
    },
    {
      "x": "1.22",
      "mean": "0.1147353244431946",
      "sd": "0.01721990554722595",
      "mask": "whole_feature_matched",
      "model": "hron"
    },
    {
      "x": "1.78",
      "mean": "0.10089654868347972",
      "sd": "0.010393664494934347",
      "mask": "whole_feature",
      "model": "tsagris"
    },
    {
      "x": "2.0",
      "mean": "0.09932792799689086",
      "sd": "0.003008733602639551",
      "mask": "random_cell",
      "model": "tsagris"
    },
    {
      "x": "2.22",
      "mean": "0.12190852140136994",
      "sd": "0.014660929677063604",
      "mask": "whole_feature_matched",
      "model": "tsagris"
    },
    {
      "x": "2.78",
      "mean": "0.10432842698496428",
      "sd": "0.009064943443233437",
      "mask": "whole_feature",
      "model": "micoda"
    },
    {
      "x": "3.0",
      "mean": "0.10688980250434132",
      "sd": "0.0024809519808008914",
      "mask": "random_cell",
      "model": "micoda"
    },
    {
      "x": "3.22",
      "mean": "0.11852194479226781",
      "sd": "0.014250900831958598",
      "mask": "whole_feature_matched",
      "model": "micoda"
    },
    {
      "x": "3.78",
      "mean": "0.031085883268336723",
      "sd": "0.005674674253809652",
      "mask": "whole_feature",
      "model": "same_day_mean"
    },
    {
      "x": "4.0",
      "mean": "0.030387937760109163",
      "sd": "0.0020470511066295505",
      "mask": "random_cell",
      "model": "same_day_mean"
    },
    {
      "x": "4.22",
      "mean": "0.03636816393376137",
      "sd": "0.005767697196924664",
      "mask": "whole_feature_matched",
      "model": "same_day_mean"
    },
    {
      "x": "4.78",
      "mean": "0.047065375317137254",
      "sd": "0.010067673063080724",
      "mask": "whole_feature",
      "model": "main"
    },
    {
      "x": "5.0",
      "mean": "0.023656077899672806",
      "sd": "0.0017781619739828797",
      "mask": "random_cell",
      "model": "main"
    },
    {
      "x": "5.22",
      "mean": "0.05812862773086801",
      "sd": "0.013607411127565057",
      "mask": "whole_feature_matched",
      "model": "main"
    },
    {
      "x": "5.78",
      "mean": "0.051560281431351926",
      "sd": "0.013980836755635134",
      "mask": "whole_feature",
      "model": "main_nometa"
    },
    {
      "x": "6.0",
      "mean": "0.023309715223343394",
      "sd": "0.002923302907810513",
      "mask": "random_cell",
      "model": "main_nometa"
    },
    {
      "x": "6.22",
      "mean": "0.06435154783818176",
      "sd": "0.020229242533790288",
      "mask": "whole_feature_matched",
      "model": "main_nometa"
    }
  ]
}

Tier 3 revision: per-seed dots copied from report_tier3/rates.csv (local U1; U3 gap) and rank_agreement_regret.csv (local U2). No predictions rescored.
{
  "tier3_means": {
    "u1_under": [
      0.5647619047619047,
      0.4394871794871795,
      0.5533333333333333
    ],
    "u1_over": [
      0.7223809523809523,
      0.7567765567765568,
      0.7652380952380952
    ],
    "u2": [
      0.8728571428571428,
      0.9400000000000001,
      0.8647619047619047
    ],
    "u3": [
      0.7042397062551327,
      0.7568378096250855,
      0.7104223179904023
    ]
  },
  "regret_good_seeds": [
    7,
    7,
    8
  ]
}