# Auto-generated — DO NOT EDIT MANUALLY
# Total: 524 tokens

ADDED_TOKENS = [

    # ── indentation and whitespace ──
    '  ',       # 2-space indent (JS/TS/YAML)
    '    ',     # 4-space indent (Python standard)
    '        ', # 8-space indent (nested blocks)
    '\t',       # tab indent
    '\n',       # line break
    '\n\n',     # paragraph / block separator
    '\n\n\n',     # section break
    '\n\n\n\n',   # larger section break
    # ── tab depth (C++/JS/Rust/Fortran) ──
    '\t\t',       # 2-tab indent
    '\t\t\t',     # 3-tab indent
    '\t\t\t\t',   # 4-tab indent

    # ── Windows line endings (if present in corpus) ──
    '\r\n',        # Windows newline
    '\r\n\r\n',    # Windows paragraph break
    # ── Comparison operators  (confirmed missing from v1 vocab) ──────────────
    "==", "!=", ">=", "<=", "===", "!==", "<=>",

    # ── Arithmetic / augmented-assignment operators ───────────────────────────
    "**", "//", "++", "--",
    "+=", "-=", "*=", "/=", "%=", "**=", "//=",
    "&=", "|=", "^=", ">>=", "<<=",
    ">>", "<<",

    # ── Arrow & membership operators ──────────────────────────────────────────
    "->", "=>", "::", "..", "...", "??", "?.", "?..",

    # ── Comment delimiters ────────────────────────────────────────────────────
    "/*", "*/", "/**", "///", "//!",


    # ── struct_bigram ──
    '_{',  # 165,812,839
    '^{',  # 97,510,718
    '}{',  # 51,457,246
    '^2',  # 42,172,936
    '_1',  # 27,117,446
    '_0',  # 22,598,138
    '_2',  # 21,416,936
    '_i',  # 14,198,817
    '_n',  # 7,611,439
    '_j',  # 6,964,149
    '_k',  # 6,757,772
    '_3',  # 6,387,750
    '_t',  # 5,372,464
    '^3',  # 4,742,044
    '^1',  # 3,259,367
    '^0',  # 2,881,100
    '^4',  # 2,852,508
    '_4',  # 2,603,663
    '}$',  # 1,224,361
    '_5',  # 1,199,518
    '_6',  # 686,278
    '^5',  # 676,665
    '^6',  # 650,136
    '_8',  # 420,056
    '$$',  # 337,328
    '_7',  # 336,532
    '^8',  # 278,844
    '^7',  # 253,588
    '${',  # 155,829
    '^9',  # 148,493
    '_9',  # 145,571

    # ── command ──
    '\\frac',  # 43,459,842
    '\\right',  # 26,357,758
    '\\left',  # 26,345,299
    '\\alpha',  # 21,746,720
    '\\mathcal',  # 20,381,439
    '\\mu',  # 19,924,633
    '\\label',  # 18,594,572
    '\\in',  # 15,344,612
    '\\lambda',  # 15,053,506
    '\\sigma',  # 13,758,517
    '\\mathrm',  # 13,278,003
    '\\rho',  # 12,702,843
    '\\gamma',  # 12,684,713
    '\\beta',  # 12,568,444
    '\\pi',  # 12,504,122
    '\\partial',  # 12,494,264
    '\\phi',  # 12,113,012
    '\\mathbf',  # 11,752,058
    '\\delta',  # 11,537,499
    '\\mathbb',  # 11,458,431
    '\\tau',  # 11,423,323
    '\\nu',  # 11,294,307
    '\\omega',  # 11,235,733
    '\\theta',  # 11,130,805
    '\\bar',  # 10,578,595
    '\\hat',  # 10,176,514
    '\\tilde',  # 10,008,742
    '\\int',  # 9,730,003
    '\\Delta',  # 9,616,913
    '\\text',  # 9,478,652
    '\\infty',  # 8,739,605
    '\\times',  # 8,669,316
    '\\sum',  # 8,644,600
    '\\Omega',  # 8,560,316
    '\\pm',  # 8,426,805
    '\\eta',  # 8,257,324
    '\\xi',  # 8,012,461
    '\\epsilon',  # 7,849,253
    '\\psi',  # 7,766,582
    '\\sqrt',  # 7,560,209
    '\\rangle',  # 7,104,911
    '\\cdot',  # 6,936,217
    '\\leq',  # 6,860,035
    '\\varepsilon',  # 6,744,360
    '\\Gamma',  # 6,132,365
    '\\ell',  # 6,049,834
    '\\nonumber',  # 5,501,955
    '\\vec',  # 5,484,362
    '\\nabla',  # 5,447,459
    '\\to',  # 5,425,943
    '\\begin',  # 5,415,174
    '\\end',  # 5,397,396
    '\\varphi',  # 5,293,104
    '\\chi',  # 5,215,501
    '\\langle',  # 5,132,025
    '\\sim',  # 5,113,705
    '\\bm',  # 5,065,166
    '\\quad',  # 4,941,639
    '\\Lambda',  # 4,786,632
    '\\kappa',  # 4,620,126
    '\\big',  # 4,507,173
    '\\Phi',  # 4,463,855
    '\\overline',  # 4,266,838
    '\\boldsymbol',  # 4,236,688
    '\\rightarrow',  # 3,636,484
    '\\Big',  # 3,618,129
    '\\prime',  # 3,613,649
    '\\dot',  # 3,457,791
    '\\geq',  # 3,173,737
    '\\zeta',  # 3,167,486
    '\\Sigma',  # 3,119,455
    '\\le',  # 2,907,873
    '\\Psi',  # 2,895,936
    '\\widetilde',  # 2,892,333
    '\\mathfrak',  # 2,752,286
    '\\qquad',  # 2,740,753
    '\\over',  # 2,564,573
    '\\otimes',  # 2,395,206
    '\\ensuremath',  # 2,368,434
    '\\approx',  # 2,342,745
    '\\sin',  # 2,307,829
    '\\circ',  # 2,306,573
    '\\cos',  # 2,299,284
    '\\equiv',  # 2,280,464
    '\\ldots',  # 2,184,862
    '\\dagger',  # 2,162,899
    '\\log',  # 2,161,598
    '\\vert',  # 2,047,942
    '\\widehat',  # 1,875,310
    '\\textrm',  # 1,816,521
    '\\ket',  # 1,767,653
    '\\cdots',  # 1,732,208
    '\\ast',  # 1,606,989
    '\\exp',  # 1,540,443
    '\\ln',  # 1,516,903
    '\\dots',  # 1,512,533
    '\\perp',  # 1,486,109
    '\\hbar',  # 1,482,037
    '\\bigg',  # 1,461,210
    '\\ge',  # 1,424,291
    '\\Pi',  # 1,419,242
    '\\neq',  # 1,413,851
    '\\Theta',  # 1,367,745
    '\\mathsf',  # 1,319,030
    '\\textbf',  # 1,312,661
    '\\simeq',  # 1,295,986
    '\\hspace',  # 1,263,774
    '\\odot',  # 1,263,010
    '\\Vert',  # 1,257,216
    '\\subset',  # 1,217,960
    '\\star',  # 1,203,609
    '\\mathscr',  # 1,181,464
    '\\underline',  # 1,113,255
    '\\lesssim',  # 1,109,230
    '\\cap',  # 1,043,889
    '\\limits',  # 1,041,844
    '\\displaystyle',  # 941,956
    '\\wedge',  # 929,665
    '\\prod',  # 919,515
    '\\lim',  # 887,617
    '\\dfrac',  # 869,027
    '\\max',  # 819,045
    '\\ll',  # 795,964
    '\\operatorname',  # 789,177
    '\\forall',  # 765,065
    '\\tfrac',  # 759,493
    '\\ref',  # 718,653
    '\\propto',  # 704,705
    '\\vartheta',  # 690,957
    '\\notag',  # 664,586
    '\\oplus',  # 650,847
    '\\sup',  # 650,696
    '\\setminus',  # 645,740
    '\\min',  # 641,513
    '\\bigl',  # 624,962
    '\\bigr',  # 617,010
    '\\dag',  # 598,696
    '\\Xi',  # 596,053
    '\\parallel',  # 588,727
    '\\gg',  # 587,371
    '\\uparrow',  # 574,768
    '\\varrho',  # 556,279
    '\\mapsto',  # 546,645
    '\\downarrow',  # 543,913
    '\\cup',  # 538,825
    '\\bullet',  # 504,215
    '\\Bigg',  # 484,218
    '\\mid',  # 468,303
    '\\scriptscriptstyle',  # 428,113
    '\\top',  # 418,102
    '\\Bigl',  # 401,198
    '\\Bigr',  # 393,753
    '\\Upsilon',  # 380,628
    '\\det',  # 374,396
    '\\iota',  # 360,875
    '\\gtrsim',  # 355,243
    '\\check',  # 339,007
    '\\subseteq',  # 338,802
    '\\tan',  # 337,620
    '\\eqref',  # 332,062
    '\\cite',  # 320,367
    '\\mp',  # 320,073
    '\\ne',  # 319,095
    '\\bra',  # 317,357
    '\\colon',  # 305,857
    '\\stackrel',  # 304,290
    '\\sinh',  # 303,914
    '\\cosh',  # 298,896
    '\\biggl',  # 294,353
    '\\not',  # 289,657
    '\\longrightarrow',  # 287,838
    '\\biggr',  # 286,373
    '\\ddot',  # 283,699
    '\\textnormal',  # 282,515
    '\\leqslant',  # 279,854
    '\\pmb',  # 273,168
    '\\mathds',  # 268,855
    '\\mathbbm',  # 266,368
    '\\Re',  # 252,532
    '\\inf',  # 242,564
    '\\vdots',  # 239,416
    '\\overset',  # 238,480
    '\\bot',  # 232,073
    '\\textit',  # 224,392
    '\\emptyset',  # 223,754
    '\\leftrightarrow',  # 215,292
    '\\textstyle',  # 214,130
    '\\mathtt',  # 207,687
    '\\cong',  # 207,174
    '\\hspace*',  # 206,962
    '\\substack',  # 203,300
    '\\tr',  # 202,449
    '\\underset',  # 201,520
    '\\backslash',  # 200,966
    '\\lbrace',  # 199,751
    '\\varpi',  # 198,222
    '\\nolimits',  # 188,232
    '\\vee',  # 187,437
    '\\scriptstyle',  # 187,160
    '\\upsilon',  # 186,575
    '\\rbrace',  # 185,626
    '\\Im',  # 178,249
    '\\mathit',  # 178,166
    '\\textsc',  # 177,656
    '\\braket',  # 177,404
    '\\dim',  # 176,613
    '\\Rightarrow',  # 174,385
    '\\mathring',  # 171,734
    '\\underbrace',  # 169,545
    '\\lvert',  # 168,870
    '\\triangle',  # 164,067
    '\\rvert',  # 162,612
    '\\square',  # 158,885
    '\\imath',  # 157,479
    '\\sharp',  # 155,986
    '\\varsigma',  # 149,268
    '\\coloneqq',  # 147,419
    '\\tanh',  # 143,658
    '\\Tr',  # 141,915
    '\\lfloor',  # 139,289
    '\\rfloor',  # 138,147
    '\\overrightarrow',  # 135,473
    '\\leftarrow',  # 135,269
    '\\notin',  # 130,352
    '\\bigcup',  # 129,544
    '\\binom',  # 129,147
    '\\geqslant',  # 129,060
    '\\lVert',  # 128,896
    '\\rVert',  # 122,292
    '\\breve',  # 118,215
    '\\oint',  # 113,376
    '\\ker',  # 107,124
    '\\iint',  # 105,267
    '\\triangleq',  # 102,297
    '\\xrightarrow',  # 96,116
    '\\ddots',  # 92,295
    '\\limsup',  # 90,046
    '\\tag',  # 86,338
    '\\div',  # 86,125
    '\\cot',  # 83,457
    '\\Biggl',  # 81,598
    '\\arg',  # 80,847
    '\\bigoplus',  # 78,499
    '\\Biggr',  # 78,433
    '\\lbrack',  # 73,382
    '\\hookrightarrow',  # 70,316
    '\\deg',  # 70,136
    '\\liminf',  # 69,974
    '\\exists',  # 68,616
    '\\supset',  # 68,006
    '\\diamond',  # 65,818
    '\\jmath',  # 65,398
    '\\flat',  # 63,959
    '\\Pr',  # 63,850
    '\\arctan',  # 63,209
    '\\wp',  # 62,917
    '\\rceil',  # 60,891
    '\\ni',  # 58,894
    '\\lceil',  # 58,808
    '\\prec',  # 56,705
    '\\Leftrightarrow',  # 54,809
    '\\rightharpoonup',  # 48,295
    '\\acute',  # 48,211
    '\\rrbracket',  # 47,644
    '\\llbracket',  # 47,586
    '\\doteq',  # 46,880
    '\\bigotimes',  # 45,672
    '\\bigcap',  # 44,708
    '\\coth',  # 44,408
    '\\Longrightarrow',  # 42,017
    '\\ketbra',  # 41,952
    '\\varnothing',  # 41,321
    '\\rbrack',  # 37,635
    '\\preceq',  # 35,680
    '\\searrow',  # 34,223
    '\\longmapsto',  # 34,136
    '\\succ',  # 32,906
    '\\succeq',  # 29,120
    '\\overleftarrow',  # 29,070
    '\\arccos',  # 27,283
    '\\bigwedge',  # 27,181
    '\\asymp',  # 26,218
    '\\varGamma',  # 25,681
    '\\varPhi',  # 25,471
    '\\varOmega',  # 25,313
    '\\angle',  # 24,263
    '\\Longleftrightarrow',  # 24,166
    '\\nearrow',  # 23,315
    '\\natural',  # 22,757
    '\\ddagger',  # 22,714
    '\\arcsin',  # 22,518
    '\\varDelta',  # 21,096
    '\\intertext',  # 20,613
    '\\Hom',  # 19,651
    '\\sec',  # 18,671
    '\\varLambda',  # 18,481
    '\\widecheck',  # 18,277
    '\\overbrace',  # 17,707
    '\\aleph',  # 16,998
    '\\End',  # 16,050
    '\\varSigma',  # 15,748
    '\\csc',  # 15,414
    '\\supseteq',  # 13,655
    '\\grave',  # 13,526

    # ── command_zero ──
    '\\Aut',  # 0
    '\\Downarrow',  # 0
    '\\Leftarrow',  # 0
    '\\Longleftarrow',  # 0
    '\\Uparrow',  # 0
    '\\bigvee',  # 0
    '\\gcd',  # 0
    '\\hom',  # 0
    '\\hookleftarrow',  # 0
    '\\iiint',  # 0
    '\\lcm',  # 0
    '\\leftharpoondown',  # 0
    '\\leftharpoonup',  # 0
    '\\longleftarrow',  # 0
    '\\measuredangle',  # 0
    '\\nexists',  # 0
    '\\nwarrow',  # 0
    '\\rightharpoondown',  # 0
    '\\subsetneq',  # 0
    '\\supsetneq',  # 0
    '\\swarrow',  # 0
    '\\triangledown',  # 0
    '\\updownarrow',  # 0
    '\\varPi',  # 0
    '\\varTheta',  # 0
    '\\varUpsilon',  # 0
    '\\varXi',  # 0
    '\\xleftarrow',  # 0

    # ── environment ──
    '\\begin{array}',  # 1,546,406
    '\\end{array}',  # 1,530,814
    '\\end{split}',  # 902,353
    '\\begin{split}',  # 901,040
    '\\end{aligned}',  # 659,646
    '\\begin{aligned}',  # 658,863
    '\\begin{pmatrix}',  # 524,979
    '\\end{pmatrix}',  # 524,680
    '\\end{cases}',  # 363,002
    '\\begin{cases}',  # 360,871
    '\\begin{bmatrix}',  # 302,420
    '\\end{bmatrix}',  # 302,245
    '\\begin{equation}',  # 289,445
    '\\end{equation}',  # 147,738
    '\\end{matrix}',  # 98,365
    '\\begin{matrix}',  # 98,234
    '\\begin{equation*}',  # 69,830
    '\\begin{eqnarray}',  # 63,408
    '\\end{tikzpicture}',  # 55,298
    '\\begin{tikzpicture}',  # 53,810
    '\\begin{align}',  # 41,955
    '\\end{gathered}',  # 41,148
    '\\begin{gathered}',  # 41,055
    '\\end{align}',  # 38,132
    '\\end{smallmatrix}',  # 30,061
    '\\begin{smallmatrix}',  # 29,895
    '\\end{eqnarray}',  # 26,318
    '\\end{align*}',  # 22,819
    '\\begin{align*}',  # 21,616
    '\\begin{eqnarray*}',  # 20,910
    '\\end{equation*}',  # 18,559
    '\\begin{proof}',  # 16,188
    '\\end{proof}',  # 15,729
    '\\end{tikzcd}',  # 11,411
    '\\begin{tikzcd}',  # 11,325
    '\\end{eqnarray*}',  # 10,060
    '\\begin{lemma}',  # 9,154
    '\\end{lemma}',  # 8,453
    '\\end{dcases}',  # 8,432
    '\\begin{dcases}',  # 8,409
    '\\begin{theorem}',  # 8,116
    '\\end{theorem}',  # 7,560
    '\\end{alignedat}',  # 6,724
    '\\begin{alignedat}',  # 6,701
    '\\end{vmatrix}',  # 6,536
    '\\begin{vmatrix}',  # 6,532
    '\\begin{remark}',  # 6,430
    '\\end{remark}',  # 6,202
    '\\end{subarray}',  # 5,869
    '\\begin{subarray}',  # 5,867
    '\\begin{Bmatrix}',  # 5,516
    '\\end{Bmatrix}',  # 5,439
    '\\begin{proposition}',  # 5,257
    '\\end{subequations}',  # 5,153
    '\\end{multlined}',  # 5,148
    '\\begin{multlined}',  # 5,136
    '\\end{proposition}',  # 4,845
    '\\begin{subequations}',  # 4,698
    '\\begin{definition}',  # 4,462
    '\\end{definition}',  # 4,292
    '\\end{CD}',  # 4,188
    '\\begin{CD}',  # 4,175
    '\\begin{gather}',  # 2,993
    '\\end{gather}',  # 2,399
    '\\begin{multline}',  # 2,374
    '\\begin{corollary}',  # 2,233
    '\\end{multline}',  # 2,200
    '\\end{corollary}',  # 2,017
    '\\begin{displaymath}',  # 1,823
    '\\begin{gather*}',  # 1,816
    '\\end{gather*}',  # 1,815
    '\\end{displaymath}',  # 1,758
    '\\end{psmallmatrix}',  # 1,655
    '\\begin{psmallmatrix}',  # 1,642
    '\\begin{example}',  # 1,377
    '\\end{pmatrix*}',  # 1,353
    '\\begin{pmatrix*}',  # 1,344
    '\\begin{bsmallmatrix}',  # 1,318
    '\\end{bsmallmatrix}',  # 1,318
    '\\end{example}',  # 1,308
    '\\end{multline*}',  # 1,210
    '\\begin{multline*}',  # 1,158
    '\\end{bmatrix*}',  # 1,031
    '\\begin{bmatrix*}',  # 1,025
    '\\end{dcases*}',  # 886
    '\\begin{dcases*}',  # 885
    '\\begin{Vmatrix}',  # 804
    '\\end{Vmatrix}',  # 803
    '\\end{cases*}',  # 628
    '\\begin{cases*}',  # 627
    '\\end{flalign}',  # 194
    '\\begin{flalign}',  # 175
    '\\begin{conjecture}',  # 90
    '\\end{conjecture}',  # 84
    '\\end{flalign*}',  # 83
    '\\begin{flalign*}',  # 75

    # ── cmd_brace ──
    '\\frac{',  # 41,066,590  # ratio=0.94
    '\\label{',  # 18,572,561  # ratio=1.00
    '\\mathcal{',  # 16,502,355  # ratio=0.81
    '\\mathrm{',  # 12,715,957  # ratio=0.96
    '\\mathbf{',  # 10,336,388  # ratio=0.88
    '\\text{',  # 9,332,584  # ratio=0.98
    '\\mathbb{',  # 8,798,968  # ratio=0.77
    '\\hat{',  # 7,263,203  # ratio=0.71
    '\\sqrt{',  # 7,123,195  # ratio=0.94
    '\\tilde{',  # 6,708,793  # ratio=0.67
    '\\bar{',  # 6,534,890  # ratio=0.62
    '\\begin{',  # 5,405,595  # ratio=1.00
    '\\end{',  # 5,383,685  # ratio=1.00
    '\\vec{',  # 3,926,326  # ratio=0.72
    '\\overline{',  # 3,597,458  # ratio=0.84
    '\\bm{',  # 3,485,363  # ratio=0.69
    '\\boldsymbol{',  # 3,417,935  # ratio=0.81
    '\\ensuremath{',  # 2,325,462  # ratio=0.98
    '\\mathfrak{',  # 2,211,203  # ratio=0.80
    '\\widetilde{',  # 2,176,411  # ratio=0.75
    '\\textrm{',  # 1,781,737  # ratio=0.98
    '\\ket{',  # 1,696,363  # ratio=0.96
    '\\widehat{',  # 1,452,074  # ratio=0.77
    '\\hspace{',  # 1,258,712  # ratio=1.00
    '\\textbf{',  # 1,257,136  # ratio=0.96
    '\\mathsf{',  # 1,143,711  # ratio=0.87
    '\\mathscr{',  # 950,488  # ratio=0.80
    '\\underline{',  # 930,908  # ratio=0.84
    '\\dfrac{',  # 850,259  # ratio=0.98
    '\\operatorname{',  # 775,966  # ratio=0.98
    '\\ref{',  # 715,432  # ratio=1.00
    '\\tfrac{',  # 621,935  # ratio=0.82
    '\\eqref{',  # 330,926  # ratio=1.00
    '\\bra{',  # 299,963  # ratio=0.95
    '\\cite{',  # 298,531  # ratio=0.93
    '\\stackrel{',  # 292,046  # ratio=0.96
    '\\textnormal{',  # 277,673  # ratio=0.98
    '\\check{',  # 241,166  # ratio=0.71
    '\\mathds{',  # 236,464  # ratio=0.88
    '\\overset{',  # 231,948  # ratio=0.97
    '\\textit{',  # 216,794  # ratio=0.97
    '\\mathbbm{',  # 215,474  # ratio=0.81
    '\\ddot{',  # 212,542  # ratio=0.75
    '\\pmb{',  # 207,457  # ratio=0.76
    '\\hspace*{',  # 206,945  # ratio=1.00
    '\\substack{',  # 201,360  # ratio=0.99
    '\\underset{',  # 198,078  # ratio=0.98
    '\\mathtt{',  # 177,480  # ratio=0.85
    '\\braket{',  # 175,722  # ratio=0.99
    '\\textsc{',  # 169,787  # ratio=0.96
    '\\mathit{',  # 167,545  # ratio=0.94
    '\\underbrace{',  # 166,365  # ratio=0.98
    '\\mathring{',  # 142,586  # ratio=0.83
    '\\overrightarrow{',  # 125,203  # ratio=0.92
    '\\binom{',  # 123,314  # ratio=0.95
    '\\breve{',  # 86,147  # ratio=0.73
    '\\tag{',  # 85,793  # ratio=0.99
    '\\xrightarrow{',  # 73,084  # ratio=0.76
    '\\acute{',  # 43,531  # ratio=0.90
    '\\ketbra{',  # 40,096  # ratio=0.96
    '\\overleftarrow{',  # 25,923  # ratio=0.89
    '\\intertext{',  # 20,563  # ratio=1.00
    '\\overbrace{',  # 17,233  # ratio=0.97
    '\\widecheck{',  # 15,605  # ratio=0.85
    '\\grave{',  # 12,090  # ratio=0.89

    # ── space_prefixed_commands (top 30) ──
    ' \\frac',
    ' \\right',
    ' \\left',
    ' \\alpha',
    ' \\mathcal',
    ' \\mu',
    ' \\label',
    ' \\in',
    ' \\lambda',
    ' \\sigma',
    ' \\mathrm',
    ' \\rho',
    ' \\gamma',
    ' \\beta',
    ' \\pi',
    ' \\partial',
    ' \\phi',
    ' \\mathbf',
    ' \\delta',
    ' \\mathbb',
    ' \\tau',
    ' \\nu',
    ' \\omega',
    ' \\theta',
    ' \\bar',
    ' \\hat',
    ' \\tilde',
    ' \\int',
    ' \\Delta',
    ' \\text',
    # Byte Fallback for OOV
    # *[f'<0x{i:02X}>' for i in range(256)],

]
