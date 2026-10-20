# Report 1 — Data audit

Amazon Shopping Queries Dataset (ESCI), small version, US locale (Reddy et al., 2022, Apache-2.0).

## 1. Size

| split | queries | pairs | products | cand_min | cand_median | cand_max |
|---|---|---|---|---|---|---|
| train | 20888 | 419653 | 351961 | 8 | 16.0 | 188 |
| test | 8956 | 181701 | 164900 | 8 | 16.0 | 95 |
| total | 29844 | 601354 | 482105 | 8 | 16.0 | 188 |

Expected sizes from spec §2.1 matched: `True`.

![candidates](figures/candidates_per_query.png)

## 2. Labels

Share of pairs per label:

| split | E | S | C | I |
|---|---|---|---|---|
| test | 0.439 | 0.350 | 0.045 | 0.167 |
| train | 0.433 | 0.352 | 0.045 | 0.169 |
| total | 0.435 | 0.351 | 0.045 | 0.169 |

Share of Exact per query: train: mean 0.430, median 0.400, queries with all candidates Exact: 0, with no Exact: 0; test: mean 0.434, median 0.429, queries with all candidates Exact: 0, with no Exact: 1

Queries with no E/S/C candidate (ideal DCG = 0, excluded from metric averages): **0**.

![share exact](figures/share_exact_per_query.png)

Most candidates are Exact, so even a random ordering puts relevant products near the top: random nDCG will be high and gains must be read relative to random and BM25.

## 3. Missing fields (empty after cleaning)

| field | train | test |
|---|---|---|
| title | 0.0% | 0.0% |
| description | 49.9% | 50.0% |
| bullets | 11.6% | 11.8% |
| brand | 4.9% | 5.1% |
| colour | 30.8% | 30.7% |

## 4. Brands

Across all pairs, the product's brand appears in the query text for **9.9%** of pairs.

| brand | products | pairs | share_pairs_brand_in_query |
|---|---|---|---|
| Nike | 2818 | 3317 | 0.607 |
| adidas | 1841 | 2191 | 0.349 |
| Amazon Essentials | 1556 | 1763 | 0.018 |
| Hanes | 1352 | 1811 | 0.088 |
| Under Armour | 1309 | 1510 | 0.376 |
| Disney | 1270 | 1502 | 0.109 |
| Amazon Basics | 1054 | 1759 | 0.007 |
| Rubie's | 916 | 1116 | 0.000 |
| Skechers | 817 | 960 | 0.174 |
| Champion | 791 | 941 | 0.365 |
| Apple | 749 | 1900 | 0.251 |
| LEGO | 734 | 1259 | 0.565 |
| Carhartt | 731 | 915 | 0.356 |
| Columbia | 709 | 852 | 0.311 |
| Samsung Electronics | 699 | 1375 | 0.584 |
| Simple Joys by Carter's | 656 | 915 | 0.000 |
| Levi's | 648 | 766 | 0.225 |
| Crocs | 641 | 779 | 0.420 |
| Unknown | 632 | 692 | 0.000 |
| Calvin Klein | 629 | 691 | 0.192 |

## 5. Queries

Query length: mean 3.92 tokens, median 4, max 29. 20.9% of queries contain a digit; 8.6% contain a negation cue.

![query length](figures/query_length.png)

Examples (training queries, random):

- numeric: `1% hydrocortisone cream without aloe` · `logitech g933` · `samsung galaxy note 10 plus otter box` · `iron man mark 85` · `body pillow cover 21 x 54` · `boys 8 nightgown` · `1-cup pyrex without lid` · `otter box for iphone 10`
- negation: `butterfly knife trainer without latch` · `outdoor ceiling fan without lights` · `do not open this book` · `50 inch tv not smart` · `06 titan trans oil cooler line not adapter` · `rolling pin without handles` · `desk chair without wheels cheap` · `1 in curling iron without clamp`
- brand: `strathmore 300 watercolor` · `100” cutain rod without center support` · `brown queen flat sheet` · `snug headphones kids noise cancelling` · `light blue toddler suits` · `healing depression without medication` · `theworldofpatio` · `5.11 moab`
- colour: `black wool dryer balls` · `gren and yelow and white balloons` · `royal blue top girl 2t` · `retinol eye cream with caffeine` · `white and pink crib sheets` · `womens sexy bodycon dress blue` · `flat tummy cream for women` · `pink sweaters with hearts`

Query sources (from `sources.csv`; metadata only, not used as a feature): other 24,107, parse_pattern 2,558, negations 2,318, behavioral 762, nlqec 99.

## 6. Join and duplicates

- `examples` joined to `products` on (`product_locale`, `product_id`): coverage **100.00%** (0 pairs without a product row).
- Duplicate (query_id, product_id) pairs: **0** (would be dropped keeping the first).

## 7. Cross-split integrity (spec §4)

- **Shared products (rule 4.2):** 25.6% of test pairs have a product that also appears in training (21.1% of unique test products). Allowed (shared catalog); no feature uses labels, so no label can leak through them.
- **Near-duplicate test queries (rule 4.3):** 26 exact matches after normalisation and 100 with char-3-gram TF-IDF cosine ≥ 0.90 (TF-IDF fit on training queries); **101** flagged in total (1.1% of test queries). Final results are reported with and without them.

Examples of flagged test queries: `* batteries not included` (cos 0.92) · `- hanging do not enter` (cos 0.90) · `-1 3/4'' center pop up drain without overflow` (cos 1.00) · `.5 inch suction cups without hooks` (cos 0.90) · `.50 cc syringe without needle` (cos 0.92) · `0.5ml syringe without needle` (cos 1.00) · `05 syringe without needle` (cos 0.91) · `1 1/4” sink drain without overflow` (cos 1.00)

## 8. Ten example queries (training split)

**ducky one two mini white** (16 candidates)

- `E` RK ROYAL KLUDGE RK61 2.4Ghz Wireless/Bluetooth/Wired 60% Mechanical Keyboard, 61 Keys RGB  — *RK ROYAL KLUDGE*
- `S` Ducky One 2 Mini RGB LED 60% Double Shot PBT Mechanical Keyboard (Cherry MX Red) — *Ducky*
- `S` Ducky One Mini RGB LED 60% Double Shot PBT Mechanical Keyboard (Cherry MX Silver) — *Ducky*
- `E` GK61 Mechanical Gaming Keyboard - 61 Keys Multi Color RGB Illuminated LED Backlit Wired Pr — *HK GAMING*
- `E` Ducky One 2 Mini Pure White - 60% RGB - Cherry MX Silent Red Switches (Cherry MX Silent Re — *Ducky*
- `S` ANNE PRO 2, 60% Wired/Wireless Mechanical Keyboard (Gateron Blue Switch/Black Case) - Full — *ANNE PRO*
- `I` Ducky One 2 Mini Mecha (Cherry MX Brown) — *Ducky*
- `S` Logitech G915 TKL Tenkeyless Lightspeed Wireless RGB Mechanical Gaming Keyboard, Low Profi — *Logitech G*
- … 8 more

**industrial fan 12** (40 candidates)

- `S` Air King 9314 14-Inch Industrial Grade High Velocity Multi Mount Fan,Black — *Air King*
- `E` TPI Corporation U12-TE Industrial Workstation Fan, Mountable, Single Phase, 12" Diameter,  — *TPI*
- `E` TPI - F-12-TE Corporation F12-TE Industrial Workstation Floor Fan, Single Phase, 12" Diame — *TPI*
- `E` Optimus F-4122 Industrial Grade High Velocity Fan Chrome Grill, 12" — *Optimus*
- `S` Air King 9312 Powder-Coated Steel Multi-Mount Wall Fan, Black — *Air King*
- `S` Westinghouse Lighting Westinghouse 7812700 Jax, Modern Industrial Style Ceiling Fan and Wa — *Westinghouse Lighting*
- `S` Air King 9020 1/6 HP Industrial Grade Wall Mount Fan, 20-Inch,Black — *Air King*
- `E` Air King 9212 12-Inch Industrial Grade High Velocity Pivoting Floor Fan — *Air King*
- … 32 more

**trolling motor mount** (16 candidates)

- `C` MotorGuide MGA015PB6 Mounting Isolator Kit — Absorb Shocks, Protect Electronics — Includes — *MotorGuide*
- `S` Minn Kota 1854021 MKA-21 Composite Quick Release Bracket (Terrova and PD Motors) — *Minn Kota*
- `C` Gamakatsu 1854017 MinnKota RTA 17 Riptide Composite Quick Release Bracket (ST and SP Motor — *Minn Kota*
- `S` Intex Motor Mount Kit for Intex inflatable Boats — *Intex*
- `E` Watersnake T24 ASP - 24 Pound Thrust 12v Electric Saltwater Trolling Motor with Transom Mo — *Watersnake*
- `E` Newport Vessels NV-Series 55lb Thrust Saltwater Transom Mounted Trolling Electric Trolling — *Newport Vessels*
- `E` ProControll Trolling Motor Mount Plate Rustproof Quick Release Upper Clamp Knobs — *EZMOUNT*
- `E` RAM Mounts RAP-400-202U Tough-Claw Trolling Motor Stabilizer with Medium Arm — *RAM MOUNTS*
- … 8 more

**8x11 rugs for living room** (16 candidates)

- `S` Home Dynamix Vega Modern Area Rug, Geometric Black/Brown/Red 7'8" X 10'4" Rectangle — *Home Dynamix*
- `S` SAFAVIEH Milan Shag Collection SG180 Solid Non-Shedding Living Room Bedroom Dining Room En — *Safavieh*
- `S` SAFAVIEH Adirondack Collection ADR108B Oriental Medallion Non-Shedding Living Room Bedroom — *Safavieh*
- `S` SAFAVIEH Dallas Shag Collection SGD257G Trellis Non-Shedding Living Room Bedroom Dining Ro — *Safavieh*
- `E` Unique Loom Solo Solid Shag Collection Area Modern Plush Rug Lush & Soft, 8 ft x 11 ft, Cl — *Unique Loom*
- `E` Unique Loom Del Mar Collection Area Rug-Transitional Inspired with Modern Contemporary Des — *Unique Loom*
- `S` Unique Loom Sofia Collection Traditional Vintage Gray Area Rug (9' x 12') — *Unique Loom*
- `S` SAFAVIEH Madison Collection MAD600C Boho Chic Glam Paisley Non-Shedding Living Room Bedroo — *Safavieh*
- … 8 more

**fleece** (16 candidates)

- `I` Vanishing Fleece: Adventures in American Wool — *Harry N. Abrams*
- `E` Columbia womens Benton Springs Full Zip Fleece Jacket, Charcoal Heather, Medium US — *Columbia*
- `E` Columbia Women's Standard Benton Springs Soft Fleece Vest, Black, Medium — *Columbia*
- `E` Columbia Men's Standard Steens Mountain 2.0 Full Zip Fleece Jacket, Charcoal Heather, Larg — *Columbia*
- `E` Columbia Women's Plus Size Fast Trek II Jacket, Black, 2X — *Columbia*
- `E` MarJunSep 5 Packs 5 Colors Lovely Pet Paw Prints Fleece Blankets for Dogs Cats Small Pets  — *MarJunSep*
- `E` Gildan Men's -Fleece Hooded -Sweatshirt, Style G18500, Dark Heather, X-Large — *-Gildan*
- `E` True North by Sleep Philosophy Micro Fleece Bed Sheet Set, Warm, Soft Plush Sheets with 14 — *True North by Sleep Philosophy*
- … 8 more

**gold necklace for girlfriend** (15 candidates)

- `C` Peora Aquamarine with Genuine Diamond Pendant in 14 Karat White Gold, Heart Shape Solitair — *Peora*
- `S` 18K Yellow Gold over Sterling Silver Diamond Heart Pendant Necklace (1/4 cttw), 18" — *Amazon Collection*
- `S` kate spade new york "Infinity and Beyond" Clear/Rose Gold Knot Mini Pendant Necklace, 17"  — *Kate Spade New York*
- `S` Customizable Your Name Bar Necklace Gift Personalized gift - Custom Jewelry Plated in 16k  — *Petite Boutique*
- `E` 1 Carat 14K Yellow Gold Round Diamond Bezel Solitaire Pendant Necklace J Color I2 Clarity — *Houston Diamond District*
- `C` DeFaith Real Rose Pendant Necklace 24K Gold Dipped, Best Gifts for her Wife Girlfriend Mot — *DEFAITH*
- `E` Gold Plated Silver I Love You Necklace Loving Gift Inscribed in 120 Languages in Pure Gold — *Nano Jewelry*
- `S` Heart Necklace 925 Sterling Silver I Love You Forever Pendant Necklace with Blue Swarovski — *AOBOCO*
- … 7 more

**vacation tops for women beach** (16 candidates)

- `I` Havaianas Women's Slim Flip Flop Sandal, Black, 7-8 — *Havaianas*
- `I` Body Glove Women's Smoothies Solo Solid Underwire Bikini Top Swimsuit, Black, E Cup — *Body Glove*
- `E` Nlife Womens Tank Tops Novelty Graphic Tees Shirt with Sayings Casual Tshirts Summer Ladie — *Nlife*
- `I` CUPSHE Women Swimsuits Sexy Bikini Sets Striped Underwear Scalloped Trim Top Beach Swimwea — *CUPSHE*
- `I` Plus Size Black Dress for Women Casual Racerback Summer Beach Sundress with Pockets Black  — *LuckyMore*
- `I` CUPSHE Women's Stripe Back Braided Straps Reversible Tank Top Strappy Lace Up Bikini Sets  — *CUPSHE*
- `E` Chulianyouhuo Resting Beach Face Tank Women Funny Graphic Print Sleeveless Beach Vacation  — *Chulianyouhuo*
- `E` LouKeith Womens Tops Sleeveless Halter Racerback Summer Casual Shirts Basic Tee Shirts Cam — *LouKeith*
- … 8 more

**women samsung smart watch** (16 candidates)

- `E` Samsung Gear S Smartwatch, White 4GB (AT&T) — *Samsung Electronics*
- `E` Samsung Gear Fit2- Pink, Small — *Samsung Electronics*
- `E` Samsung Gear Sport Smartwatch (Bluetooth), Blue, SM-R600NZBAXAR – US Version with Warranty — *Samsung Electronics*
- `E` Samsung Galaxy Watch (42mm, GPS, Bluetooth) – Rose Gold (US Version) — *Samsung Electronics*
- `S` Fossil Women's Gen 4 Sport Heart Rate Metal and Silicone Touchscreen Smartwatch, Color:Blu — *Fossil*
- `E` SAMSUNG Galaxy Watch (42mm) Rose Gold (Bluetooth) (Renewed) — *Samsung Electronics*
- `E` SAMSUNG Galaxy Watch Active (40MM, GPS, Bluetooth) Smart Watch with Fitness Tracking, and  — *SAMSUNG*
- `E` Samsung Galaxy Watch Active - 40mm, IP68 Water Resistant, Wireless Charging, SM-R500N Inte — *Samsung Electronics*
- … 8 more

**red athletic shorts women** (16 candidates)

- `E` Original Soffe Cheer Shorts, Red, Adult Large — *Cheerleading Company*
- `E` Augusta Sportswear Women's Junior Fit Pulse Team Short, Red/White/Black, Small — *Augusta Sportswear*
- `E` Champion Men's Long Mesh Short with Pockets,Crimson,XX-Large — *Champion*
- `E` Nike Women's Dri-fit Tempo Track 3.5 Short (Sport Red/White/Black/Wolf Grey, Small) — *Nike*
- `E` Nike Girls Dry Tempo Running Shorts (X-Large, Red/White) — *Nike*
- `E` adidas Women's Standard Parma 16 Shorts, Power Red/White, Small — *adidas*
- `E` Champion Women's Mesh Short, Scarlet, Medium — *Champion*
- `S` Clothe Co. Womens Running Shorts Moisture Wicking Sport Running Shorts, Black/True Red/Whi — *Clothe Co.*
- … 8 more

**flush mount ceiling fans without lights** (41 candidates)

- `E` Emerson Ceiling Fans CF905ORB Prima Snugger 52-Inch Low Profile Ceiling Fan With Wall Cont — *Emerson*
- `E` Craftmade 3 Blade White Ceiling Fan Without Light VE58W3 Velocity 58 Inch and Wall Control — *Craftmade*
- `E` Hunter Fan Company 51059 Hunter Indoor Low Profile IV Ceiling Fan with Pull Chain Control, — *Hunter Fan Company*
- `E` Hunter Indoor Low Profile III Ceiling Fan with Pull Chain Control — *Hunter Fan Company*
- `E` Hunter Sea Wind Indoor / Outdoor Ceiling Fan with Pull Chain Control, 48", White — *Hunter Fan Company*
- `S` Honeywell 50182 Quick-2-Hang Hugger Ceiling Fan, 52” Dimmable LED White Swirled Marble Fix — *Honeywell Ceiling Fans*
- `E` 52" Casa Delta-Wing Modern Contemporary Farmhouse Indoor Outdoor Ceiling Fan with Remote S — *Casa Vieja*
- `E` Hunter Fan Company 59247 Dempsey Indoor Low Profile Ceiling Fan with Remote Control, 52",  — *Hunter Fan Company*
- … 33 more

`notebooks/01_eda.ipynb` reproduces the main figures.
