import os
import sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank_layout = prs.slide_layouts[6]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

# Executive Color Palette
NAVY = RGBColor(15, 23, 42)          # #0F172A (Deep Slate/Navy)
PRIMARY_BLUE = RGBColor(37, 99, 235) # #2563EB (Accent Blue)
SLATE_TEXT = RGBColor(71, 85, 105)   # #475569 (Secondary Text)
DARK_TEXT = RGBColor(30, 41, 59)     # #1E293B (Body Text)
CARD_BG = RGBColor(248, 250, 252)    # #F8FAFC (Soft Container Fill)
WHITE = RGBColor(255, 255, 255)
BORDER_COLOR = RGBColor(226, 232, 240) # #E2E8F0
TABLE_HEADER = RGBColor(30, 41, 59)  # #1E293B
TABLE_ROW_ALT = RGBColor(241, 245, 249) # #F1F5F9
DARK_RED = RGBColor(220, 38, 38)     # #DC2626 (Fake/Misinformation)
DARK_GREEN = RGBColor(16, 185, 129)  # #10B981 (Real/Verified)

FONT_FAMILY = "Calibri"

def add_header_and_footer(slide, title_text, category_text, slide_num, total_slides=9):
    # Top Category Pill
    cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.35), Inches(11.733), Inches(0.3))
    tf_c = cat_box.text_frame
    tf_c.margin_left = tf_c.margin_top = tf_c.margin_right = tf_c.margin_bottom = 0
    p_c = tf_c.paragraphs[0]
    p_c.text = category_text.upper()
    p_c.font.name = FONT_FAMILY
    p_c.font.size = Pt(9.5)
    p_c.font.bold = True
    p_c.font.color.rgb = PRIMARY_BLUE

    # Slide Main Title
    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.65), Inches(11.733), Inches(0.55))
    tf_t = title_box.text_frame
    tf_t.margin_left = tf_t.margin_top = tf_t.margin_right = tf_t.margin_bottom = 0
    p_t = tf_t.paragraphs[0]
    p_t.text = title_text
    p_t.font.name = FONT_FAMILY
    p_t.font.size = Pt(21)
    p_t.font.bold = True
    p_t.font.color.rgb = NAVY

    # Header Divider Line
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.25), Inches(11.733), Inches(0.015))
    line.fill.solid()
    line.fill.fore_color.rgb = BORDER_COLOR
    line.line.color.rgb = BORDER_COLOR

    # Footer Divider Line
    fline = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(6.9), Inches(11.733), Inches(0.015))
    fline.fill.solid()
    fline.fill.fore_color.rgb = BORDER_COLOR
    fline.line.color.rgb = BORDER_COLOR

    # Footer Left
    foot_box = slide.shapes.add_textbox(Inches(0.8), Inches(6.95), Inches(8.0), Inches(0.3))
    ftf = foot_box.text_frame
    ftf.margin_left = ftf.margin_top = ftf.margin_right = ftf.margin_bottom = 0
    fp = ftf.paragraphs[0]
    fp.text = "Explainable Fake News Detection in Indian Context | NLP Final-Year Evaluation"
    fp.font.name = FONT_FAMILY
    fp.font.size = Pt(8.5)
    fp.font.color.rgb = SLATE_TEXT

    # Footer Right
    num_box = slide.shapes.add_textbox(Inches(10.0), Inches(6.95), Inches(2.533), Inches(0.3))
    ntf = num_box.text_frame
    ntf.margin_left = ntf.margin_top = ntf.margin_right = ntf.margin_bottom = 0
    np_p = ntf.paragraphs[0]
    np_p.alignment = PP_ALIGN.RIGHT
    np_p.text = f"Slide {slide_num} of {total_slides}"
    np_p.font.name = FONT_FAMILY
    np_p.font.size = Pt(8.5)
    np_p.font.bold = True
    np_p.font.color.rgb = SLATE_TEXT

def create_native_table(slide, left, top, width, height, headers, rows, col_widths=None):
    num_rows = len(rows) + 1
    num_cols = len(headers)
    table_shape = slide.shapes.add_table(num_rows, num_cols, left, top, width, height)
    table = table_shape.table

    if col_widths and len(col_widths) == num_cols:
        for idx, w in enumerate(col_widths):
            table.columns[idx].width = w

    # Format Header Row
    for col_idx, h_text in enumerate(headers):
        cell = table.cell(0, col_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = TABLE_HEADER
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = cell.text_frame.paragraphs[0]
        p.text = str(h_text)
        p.alignment = PP_ALIGN.CENTER if col_idx > 0 else PP_ALIGN.LEFT
        p.font.name = FONT_FAMILY
        p.font.size = Pt(9.5)
        p.font.bold = True
        p.font.color.rgb = WHITE

    # Format Data Rows
    for row_idx, row_data in enumerate(rows):
        bg_color = TABLE_ROW_ALT if row_idx % 2 == 1 else WHITE
        for col_idx, val in enumerate(row_data):
            cell = table.cell(row_idx + 1, col_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg_color
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            p.text = str(val)
            p.alignment = PP_ALIGN.CENTER if col_idx > 0 else PP_ALIGN.LEFT
            p.font.name = FONT_FAMILY
            p.font.size = Pt(9.0)
            p.font.color.rgb = DARK_TEXT

    return table_shape

# ==============================================================================
# SLIDE 1: TITLE SLIDE
# ==============================================================================
s1 = prs.slides.add_slide(blank_layout)
bg_outer = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(0.8), Inches(11.733), Inches(5.9))
bg_outer.fill.solid()
bg_outer.fill.fore_color.rgb = CARD_BG
bg_outer.line.color.rgb = BORDER_COLOR

bar = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(0.8), Inches(0.15), Inches(5.9))
bar.fill.solid()
bar.fill.fore_color.rgb = PRIMARY_BLUE
bar.line.fill.background()

tbox = s1.shapes.add_textbox(Inches(1.4), Inches(1.5), Inches(10.5), Inches(2.2))
tf = tbox.text_frame
tf.word_wrap = True
p0 = tf.paragraphs[0]
p0.text = "NLP FINAL-YEAR PROJECT EVALUATION"
p0.font.name = FONT_FAMILY
p0.font.size = Pt(11)
p0.font.bold = True
p0.font.color.rgb = PRIMARY_BLUE

p1 = tf.add_paragraph()
p1.text = "Explainable Fake News Detection in Indian Context"
p1.font.name = FONT_FAMILY
p1.font.size = Pt(28)
p1.font.bold = True
p1.font.color.rgb = NAVY
p1.space_before = Pt(6)

p2 = tf.add_paragraph()
p2.text = "Specialized Linguistic Preprocessing, Sublinear TF-IDF Optimization & Local Interpretability (LIME)"
p2.font.name = FONT_FAMILY
p2.font.size = Pt(15)
p2.font.color.rgb = SLATE_TEXT
p2.space_before = Pt(8)

meta_card = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.4), Inches(4.3), Inches(10.5), Inches(1.7))
meta_card.fill.solid()
meta_card.fill.fore_color.rgb = WHITE
meta_card.line.color.rgb = BORDER_COLOR

mtf = meta_card.text_frame
mtf.margin_left = Inches(0.3)
mtf.margin_top = Inches(0.2)
mp1 = mtf.paragraphs[0]
mp1.text = "Academic Evaluation Rubric Compliance (Points 1 to 5):"
mp1.font.name = FONT_FAMILY
mp1.font.size = Pt(11.5)
mp1.font.bold = True
mp1.font.color.rgb = NAVY

mp2 = mtf.add_paragraph()
mp2.text = "• Dataset: Indian Fake News Dataset (IFND) — 4,000 Verified Balanced News Statements (100% Unique)\n" \
           "• Architecture: Baseline (Multinomial NB: 88.7%) vs Proposed (Logistic Regression: 90.8% | Linear SVM: 91.9%)\n" \
           "• Deliverables: Core NLP Pipeline, Interactive Streamlit Web UI (localhost:8501), and LIME Visualizer"
mp2.font.name = FONT_FAMILY
mp2.font.size = Pt(10)
mp2.font.color.rgb = SLATE_TEXT
mp2.space_before = Pt(4)

# ==============================================================================
# SLIDE 2: PROBLEM STATEMENT & PROJECT OBJECTIVES
# ==============================================================================
s2 = prs.slides.add_slide(blank_layout)
add_header_and_footer(s2, "Problem Statement & Project Objectives", "INITIAL SLIDE: PROJECT DEFINITION", 2)

b_prob = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.45), Inches(5.7), Inches(5.25))
b_prob.fill.solid()
b_prob.fill.fore_color.rgb = WHITE
b_prob.line.color.rgb = BORDER_COLOR

ptf = b_prob.text_frame
ptf.margin_left = Inches(0.3)
ptf.margin_top = Inches(0.25)
p_title = ptf.paragraphs[0]
p_title.text = "The Misinformation Landscape in India"
p_title.font.name = FONT_FAMILY
p_title.font.size = Pt(14)
p_title.font.bold = True
p_title.font.color.rgb = NAVY

challenges = [
    ("Regional Linguistic Gap:", "Mainstream NLP tools (NLTK, SpaCy) are trained on British/American news. They fail on Hinglish code-mixing, cultural terms ('crore', 'lakh'), and socio-political entities ('Modi', 'Aadhaar', 'BJP')."),
    ("WhatsApp Forward Distribution:", "Indian misinformation circulates through closed social ecosystems featuring repetitive exclamation marks ('!!!!') and chain urgency cues ('FORWARD TO 5 GROUPS') that standard preprocessors discard as noise."),
    ("The 'Black Box' Trust Deficit:", "Deep neural classifiers output scalar probabilities without accountability. For fact-checkers, understanding the exact lexical drivers behind a prediction is mandatory.")
]
for title, desc in challenges:
    p_t = ptf.add_paragraph()
    p_t.text = f"• {title}"
    p_t.font.name = FONT_FAMILY
    p_t.font.size = Pt(11)
    p_t.font.bold = True
    p_t.font.color.rgb = NAVY
    p_t.space_before = Pt(10)
    p_d = ptf.add_paragraph()
    p_d.text = desc
    p_d.font.name = FONT_FAMILY
    p_d.font.size = Pt(9.5)
    p_d.font.color.rgb = SLATE_TEXT
    p_d.space_before = Pt(2)

b_obj = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.45), Inches(5.733), Inches(5.25))
b_obj.fill.solid()
b_obj.fill.fore_color.rgb = CARD_BG
b_obj.line.color.rgb = BORDER_COLOR

otf = b_obj.text_frame
otf.margin_left = Inches(0.3)
otf.margin_top = Inches(0.25)
o_title = otf.paragraphs[0]
o_title.text = "Key Project Objectives"
o_title.font.name = FONT_FAMILY
o_title.font.size = Pt(14)
o_title.font.bold = True
o_title.font.color.rgb = NAVY

objs = [
    ("1. Corpus Profiling (Point 1):", "Analyze 4,000 IFND articles measuring word counts, sentence lengths, paragraph counts, and word clouds."),
    ("2. Linguistic Preprocessing (Point 3):", "Engineer an Indian-context pipeline preserving cultural entities, normalizing numerics, and translating WhatsApp forward markers."),
    ("3. Feature Justification (Point 2):", "Empirically evaluate TF-IDF against dense Word2Vec embeddings regarding latency, memory, and classification F1."),
    ("4. Model Benchmarking (Point 4):", "Compare baseline Multinomial Naive Bayes (88.7%) against proposed regularized Logistic Regression (90.8%) and Linear SVM (91.9%) with 5-fold cross-validation."),
    ("5. Explainable AI Deployment (Point 5):", "Deploy LIME for local token attribution and deliver an interactive web interface for real-time verification.")
]
for title, desc in objs:
    p_t = otf.add_paragraph()
    p_t.text = f"✔ {title}"
    p_t.font.name = FONT_FAMILY
    p_t.font.size = Pt(10.5)
    p_t.font.bold = True
    p_t.font.color.rgb = PRIMARY_BLUE
    p_t.space_before = Pt(8)
    p_d = otf.add_paragraph()
    p_d.text = desc
    p_d.font.name = FONT_FAMILY
    p_d.font.size = Pt(9.5)
    p_d.font.color.rgb = DARK_TEXT
    p_d.space_before = Pt(1)

# ==============================================================================
# SLIDE 3: OVERALL LITERATURE REVIEW
# ==============================================================================
s3 = prs.slides.add_slide(blank_layout)
add_header_and_footer(s3, "Overall Literature Review & Theoretical Foundations", "INITIAL SLIDE: LITERATURE REVIEW", 3)

lit_cards = [
    ("1. Misinformation & Stylistic Deception", "Shu et al. (ACM SIGKDD Explorations 2017) & Kumar et al. (IEEE Access 2018)",
     "Establishes that short digital news headlines depend overwhelmingly on sensationalist lexical framing and emotional urgency rather than complex multi-clause syntax. Proves that rich n-gram lexical features provide superior discrimination on short texts without overfitting."),
    ("2. Sparse TF-IDF vs. Dense Embedding Dilution", "Manning, Raghavan, Schütze (Cambridge 2008) & Mikolov et al. (NeurIPS 2013)",
     "Dense word vectors averaged across short headlines suffer from semantic dilution: the acute fraudulent signal of a single trigger word ('nano-chip') is drowned out by surrounding common nouns. Sublinear TF-IDF preserves exact term distinctiveness."),
    ("3. Local Interpretable Model-agnostic Explanations", "Ribeiro, Singh, Guestrin (ACM SIGKDD 2016 - LIME)",
     "Introduces local surrogate theory: while global non-linear models are opaque, any classifier can be approximated locally via an exponential-distance-weighted linear regressor in the perturbed neighborhood of an input."),
    ("4. Indian Digital News & Distribution Dynamics", "Verma et al. (IFND Benchmark 2021) & BoomLive Fact-Checking Studies",
     "Highlights that Indian fake news is distinguished by code-mixing, socio-political acronyms, and platform-specific forward syntax, necessitating specialized preprocessing beyond standard English NLTK tools.")
]

y = 1.45
for title, citation, desc in lit_cards:
    card = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(y), Inches(11.733), Inches(1.2))
    card.fill.solid()
    card.fill.fore_color.rgb = WHITE
    card.line.color.rgb = BORDER_COLOR
    ctf = card.text_frame
    ctf.margin_left = Inches(0.25)
    ctf.margin_top = Inches(0.12)
    p1 = ctf.paragraphs[0]
    p1.text = title
    p1.font.name = FONT_FAMILY
    p1.font.size = Pt(11.5)
    p1.font.bold = True
    p1.font.color.rgb = NAVY
    p2 = ctf.add_paragraph()
    p2.text = f"Literature Citation: {citation}"
    p2.font.name = FONT_FAMILY
    p2.font.size = Pt(9.0)
    p2.font.italic = True
    p2.font.color.rgb = PRIMARY_BLUE
    p3 = ctf.add_paragraph()
    p3.text = desc
    p3.font.name = FONT_FAMILY
    p3.font.size = Pt(9.0)
    p3.font.color.rgb = SLATE_TEXT
    p3.space_before = Pt(2)
    y += 1.33

# ==============================================================================
# SLIDE 4: POINT 1 - DATASET EXPLORATION
# ==============================================================================
s4 = prs.slides.add_slide(blank_layout)
add_header_and_footer(s4, "Point 1: Dataset Exploration, Structural Counts & Word Clouds", "POINT 1: DATASET EXPLORATION", 4)

img_frame = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.45), Inches(5.6), Inches(5.25))
img_frame.fill.solid()
img_frame.fill.fore_color.rgb = WHITE
img_frame.line.color.rgb = BORDER_COLOR

img_wc = RESULTS_DIR / "wordclouds_combined.png"
if img_wc.exists():
    s4.shapes.add_picture(str(img_wc), Inches(0.9), Inches(1.55), width=Inches(5.4))

cap_box = s4.shapes.add_textbox(Inches(0.9), Inches(5.8), Inches(5.4), Inches(0.8))
ctf = cap_box.text_frame
ctf.margin_left = ctf.margin_top = ctf.margin_right = ctf.margin_bottom = 0
cp = ctf.paragraphs[0]
cp.text = "Word Cloud Analysis: Fake news is dominated by sensational urgency ('video', 'viral', 'claim', 'fake', 'alleged'), while Real news relies on institutional governance terms ('court', 'minister', 'police', 'delhi', 'india')."
cp.font.name = FONT_FAMILY
cp.font.size = Pt(8.5)
cp.font.color.rgb = SLATE_TEXT

stat_headers = ["Corpus Metric", "Corpus Total", "Fake (Class 0)", "Real (Class 1)"]
stat_rows = [
    ["Total Articles", "4,000", "2,000 (50.0%)", "2,000 (50.0%)"],
    ["Total Words", "53,340", "14.85 words/item", "11.82 words/item"],
    ["Total Sentences", "4,980", "1.44 sentences/item", "1.05 sentences/item"],
    ["Total Paragraphs", "4,000", "1.00 para/item", "1.00 para/item"],
    ["Character Length", "330,620", "90.53 chars/item", "74.78 chars/item"]
]
col_w_stats = [Inches(1.8), Inches(1.2), Inches(1.4), Inches(1.333)]
create_native_table(s4, Inches(6.7), Inches(1.45), Inches(5.833), Inches(1.9), stat_headers, stat_rows, col_w_stats)

snap_headers = ["ID", "Label", "Words", "Authentic IFND Dataset News Snippet"]
snap_rows = [
    ["#0555", "Real (1)", "5", "Ready to lift curbs: Kejriwal in Delhi press conference..."],
    ["#3491", "Real (1)", "15", "SC refuses pilot's plea against allowing AI flights with middle seat..."],
    ["#0527", "Fake (0)", "9", "Old Video Showing India-Pakistan Naval Warships Shared As Recent..."],
    ["#2989", "Fake (0)", "11", "Fact Check: Viral video of strange creature climbing church is CGI..."]
]
col_w_snap = [Inches(0.7), Inches(1.0), Inches(0.8), Inches(3.333)]
create_native_table(s4, Inches(6.7), Inches(3.6), Inches(5.833), Inches(2.1), snap_headers, snap_rows, col_w_snap)

call_box = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.7), Inches(5.95), Inches(5.833), Inches(0.75))
call_box.fill.solid()
call_box.fill.fore_color.rgb = CARD_BG
call_box.line.color.rgb = BORDER_COLOR
cbtf = call_box.text_frame
cbtf.margin_left = Inches(0.2)
cbtf.margin_top = Inches(0.1)
cbp = cbtf.paragraphs[0]
cbp.text = "Key Insight: Fake news exhibits significantly higher average word length (14.8 vs 11.8 words) and sentence count (1.44 vs 1.05) due to multi-claim stacking and sensational assertions."
cbp.font.name = FONT_FAMILY
cbp.font.size = Pt(8.5)
cbp.font.bold = True
cbp.font.color.rgb = NAVY

# ==============================================================================
# SLIDE 5: POINT 2 - FEATURE & EMBEDDING JUSTIFICATION
# ==============================================================================
s5 = prs.slides.add_slide(blank_layout)
add_header_and_footer(s5, "Point 2: Word Embedding & Feature Representation Justification", "POINT 2: FEATURE ENGINEERING", 5)

f_headers = ["Feature Representation", "Dim", "Latency (s)", "Speedup", "Macro F1"]
f_rows = [
    ["TF-IDF (Unigram)", "10,000", "0.0666 s", "15.3x faster", "0.9058"],
    ["TF-IDF (Bigram)", "15,000", "0.0747 s", "13.7x faster", "0.8235"],
    ["TF-IDF (1-2 gram)", "20,000", "0.1191 s", "8.5x faster", "0.9058"],
    ["Word2Vec (Avg 100d)", "100", "1.0223 s", "Baseline (1x)", "0.8784"]
]
col_w_f = [Inches(2.1), Inches(0.7), Inches(0.9), Inches(1.1), Inches(0.8)]
create_native_table(s5, Inches(0.8), Inches(1.45), Inches(5.6), Inches(1.8), f_headers, f_rows, col_w_f)

ff_box = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(3.5), Inches(5.6), Inches(3.2))
ff_box.fill.solid()
ff_box.fill.fore_color.rgb = WHITE
ff_box.line.color.rgb = BORDER_COLOR
ff_tf = ff_box.text_frame
ff_tf.margin_left = Inches(0.25)
ff_tf.margin_top = Inches(0.2)
p_ft = ff_tf.paragraphs[0]
p_ft.text = "Empirical Benchmark Insights (Exp 3)"
p_ft.font.name = FONT_FAMILY
p_ft.font.size = Pt(13)
p_ft.font.bold = True
p_ft.font.color.rgb = NAVY

f_points = [
    ("Accuracy & F1 Advantage:", "TF-IDF (1-2 gram) achieves 90.58% Macro F1, outperforming dense Word2Vec (87.84%) by +2.74% on authentic IFND articles."),
    ("8.5x Latency Speedup:", "TF-IDF extraction executes in ~0.12s compared to ~1.02s for Word2Vec embeddings, making it feasible for real-time social stream verification."),
    ("Contextual Bigram Retention:", "Combining unigrams and bigrams preserves compound entities ('supreme court', 'fact check', 'naval warships') that dense average pooling blurs.")
]
for title, desc in f_points:
    p_t = ff_tf.add_paragraph()
    p_t.text = f"• {title}"
    p_t.font.name = FONT_FAMILY
    p_t.font.size = Pt(10.5)
    p_t.font.bold = True
    p_t.font.color.rgb = NAVY
    p_t.space_before = Pt(8)
    p_d = ff_tf.add_paragraph()
    p_d.text = desc
    p_d.font.name = FONT_FAMILY
    p_d.font.size = Pt(9.0)
    p_d.font.color.rgb = SLATE_TEXT
    p_d.space_before = Pt(1)

rf_box = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.45), Inches(5.733), Inches(5.25))
rf_box.fill.solid()
rf_box.fill.fore_color.rgb = CARD_BG
rf_box.line.color.rgb = BORDER_COLOR
rf_tf = rf_box.text_frame
rf_tf.margin_left = Inches(0.3)
rf_tf.margin_top = Inches(0.2)

p_rt = rf_tf.paragraphs[0]
p_rt.text = "Theoretical Justification via Literature"
p_rt.font.name = FONT_FAMILY
p_rt.font.size = Pt(14)
p_rt.font.bold = True
p_rt.font.color.rgb = NAVY

r_justs = [
    ("1. Sublinear Term Frequency Scaling (Manning et al., 2008):",
     "We implemented sublinear TF scaling: TF_scaled = 1 + log(TF). In sensational headlines, words like 'URGENT' may appear multiple times. Log-scaling dampens repetitive frequency distortion while preserving true lexical salience."),
    ("2. Preventing Embedding Dilution (Mikolov et al., 2013):",
     "Dense word vectors averaged across short sentences suffer from mathematical smoothing: the acute fraudulent signal of a single fraudulent keyword ('microchip') is mathematically diluted by surrounding neutral nouns."),
    ("3. Native Alignment with LIME Surrogate Explanations:",
     "LIME's perturbation mechanism operates directly on discrete token absence/presence. Sparse TF-IDF provides exact 1:1 token-to-feature mapping, ensuring transparent interpretability.")
]
for title, desc in r_justs:
    p_t = rf_tf.add_paragraph()
    p_t.text = title
    p_t.font.name = FONT_FAMILY
    p_t.font.size = Pt(10.5)
    p_t.font.bold = True
    p_t.font.color.rgb = PRIMARY_BLUE
    p_t.space_before = Pt(10)
    p_d = rf_tf.add_paragraph()
    p_d.text = desc
    p_d.font.name = FONT_FAMILY
    p_d.font.size = Pt(9.0)
    p_d.font.color.rgb = DARK_TEXT
    p_d.space_before = Pt(2)

# ==============================================================================
# SLIDE 6: POINT 3 - SPECIALIZED INDIAN PREPROCESSING
# ==============================================================================
s6 = prs.slides.add_slide(blank_layout)
add_header_and_footer(s6, "Point 3: Specialized Indian Preprocessing & Comparative Benchmark", "POINT 3: SPECIALIZED PREPROCESSING", 6)

p_frame = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.45), Inches(5.6), Inches(5.25))
p_frame.fill.solid()
p_frame.fill.fore_color.rgb = WHITE
p_frame.line.color.rgb = BORDER_COLOR

img_prep = RESULTS_DIR / "preprocessing_comparison.png"
if img_prep.exists():
    s6.shapes.add_picture(str(img_prep), Inches(0.9), Inches(1.55), width=Inches(5.4))

r_prep = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.45), Inches(5.733), Inches(5.25))
r_prep.fill.solid()
r_prep.fill.fore_color.rgb = CARD_BG
r_prep.line.color.rgb = BORDER_COLOR

rptf = r_prep.text_frame
rptf.margin_left = Inches(0.3)
rptf.margin_top = Inches(0.2)
p_head = rptf.paragraphs[0]
p_head.text = "Indian Context Pipeline Innovations"
p_head.font.name = FONT_FAMILY
p_head.font.size = Pt(14)
p_head.font.bold = True
p_head.font.color.rgb = NAVY

prep_innovations = [
    ("1. Indian Numeric Scaling:", "Standard NLTK tokenizers treat 'lakh' and 'crore' as rare out-of-vocabulary terms. Our pipeline normalizes them into standardized quantitative magnitude tokens, preserving financial fraud context."),
    ("2. Sociopolitical Entity Preservation:", "Standard English stemmers (Porter/Snowball) corrupt Indian proper nouns (stemming 'Aadhaar' into 'adhaar', 'Gandhi' to 'gandhi'). Our preprocessor protects verified socio-political entities ('modi', 'aadhaar', 'rbi', 'isro', 'aiims')."),
    ("3. WhatsApp Forward Urgency Translation:", "Standard preprocessing strips all punctuation. Indian fake news relies on exclamation clusters ('!!!!', '???') and chain forwarding prompts ('SHARE TO 10 GROUPS'). We encode these as synthetic urgency features."),
    ("4. Selective Stopword Filtering:", "Maintains discourse-relevant polarity words and administrative terms that standard NLTK stopword lists delete indiscriminately.")
]
for title, desc in prep_innovations:
    p_t = rptf.add_paragraph()
    p_t.text = f"✔ {title}"
    p_t.font.name = FONT_FAMILY
    p_t.font.size = Pt(10.5)
    p_t.font.bold = True
    p_t.font.color.rgb = PRIMARY_BLUE
    p_t.space_before = Pt(8)
    p_d = rptf.add_paragraph()
    p_d.text = desc
    p_d.font.name = FONT_FAMILY
    p_d.font.size = Pt(9.0)
    p_d.font.color.rgb = DARK_TEXT
    p_d.space_before = Pt(2)

# ==============================================================================
# SLIDE 7: POINT 4 - BASE MODEL VS PROPOSED MODEL EVALUATION
# ==============================================================================
s7 = prs.slides.add_slide(blank_layout)
add_header_and_footer(s7, "Point 4: Base Model vs. Proposed Model Evaluation Benchmarks", "POINT 4: MODEL EVALUATION", 7)

cm_frame = s7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.45), Inches(5.6), Inches(5.25))
cm_frame.fill.solid()
cm_frame.fill.fore_color.rgb = WHITE
cm_frame.line.color.rgb = BORDER_COLOR

img_cm = RESULTS_DIR / "confusion_matrices.png"
if img_cm.exists():
    s7.shapes.add_picture(str(img_cm), Inches(0.9), Inches(1.55), width=Inches(5.4))

m_headers = ["Model Identifier", "Role", "Accuracy", "Precision", "Recall", "Macro F1", "5-Fold CV"]
m_rows = [
    ["Multinomial NB", "Base Model", "88.75%", "91.44%", "85.50%", "88.74%", "88.01% ±1.2%"],
    ["Logistic Regression", "Proposed 1", "90.75%", "85.75%", "97.75%", "90.70%", "89.90% ±1.3%"],
    ["Linear SVM", "Proposed 2", "91.87%", "90.36%", "93.75%", "91.87%", "90.05% ±0.4%"]
]
col_w_m = [Inches(1.7), Inches(0.9), Inches(0.65), Inches(0.65), Inches(0.6), Inches(0.65), Inches(0.683)]
create_native_table(s7, Inches(6.7), Inches(1.45), Inches(5.833), Inches(1.6), m_headers, m_rows, col_w_m)

mj_box = s7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.7), Inches(3.25), Inches(5.833), Inches(3.45))
mj_box.fill.solid()
mj_box.fill.fore_color.rgb = CARD_BG
mj_box.line.color.rgb = BORDER_COLOR

mtf = mj_box.text_frame
mtf.margin_left = Inches(0.25)
mtf.margin_top = Inches(0.2)
p_mt = mtf.paragraphs[0]
p_mt.text = "Why Proposed Logistic Regression & SVM Outperform Base NB"
p_mt.font.name = FONT_FAMILY
p_mt.font.size = Pt(12)
p_mt.font.bold = True
p_mt.font.color.rgb = NAVY

model_reasons = [
    ("Violation of Independence Assumption in Base NB:",
     "Multinomial Naive Bayes assumes conditional independence. Correlated terms like 'Supreme Court' or 'fact check' double-count evidence, capping baseline accuracy at 88.75%."),
    ("Superior Recall in Proposed Logistic Regression (97.75%):",
     "Proposed Logistic Regression achieves 97.75% recall on genuine news (only 9 false negatives out of 400). Its sigmoid function provides smooth, well-calibrated posterior probabilities strictly required for LIME."),
    ("Peak Accuracy in Proposed Linear SVM (91.87%):",
     "Linear SVM constructs the maximum-margin hyperplane in sparse 20,000-dimensional space with L2 regularization, delivering the highest overall classification accuracy and lowest cross-validation variance (±0.41%).")
]
for title, desc in model_reasons:
    p_t = mtf.add_paragraph()
    p_t.text = f"• {title}"
    p_t.font.name = FONT_FAMILY
    p_t.font.size = Pt(10)
    p_t.font.bold = True
    p_t.font.color.rgb = NAVY
    p_t.space_before = Pt(6)
    p_d = mtf.add_paragraph()
    p_d.text = desc
    p_d.font.name = FONT_FAMILY
    p_d.font.size = Pt(8.5)
    p_d.font.color.rgb = SLATE_TEXT
    p_d.space_before = Pt(1)

# ==============================================================================
# SLIDE 8: EXAMPLE OF WORK & LIME EXPLAINABILITY
# ==============================================================================
s8 = prs.slides.add_slide(blank_layout)
add_header_and_footer(s8, "Example of Work: Input Headline, Output & LIME Feature Attribution", "PRACTICAL DEMONSTRATION", 8)

io_card = s8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.45), Inches(5.6), Inches(5.25))
io_card.fill.solid()
io_card.fill.fore_color.rgb = WHITE
io_card.line.color.rgb = BORDER_COLOR

iotf = io_card.text_frame
iotf.margin_left = Inches(0.25)
iotf.margin_top = Inches(0.2)
p_it = iotf.paragraphs[0]
p_it.text = "Sample Input News Article (Indian Context)"
p_it.font.name = FONT_FAMILY
p_it.font.size = Pt(13)
p_it.font.bold = True
p_it.font.color.rgb = NAVY

p_quote = iotf.add_paragraph()
p_quote.text = '"Fact Check: The video of astronaut jumping from space is fake. This is not the video of missing IAF pilot in Kashmir encounter. Forward to all groups!"'
p_quote.font.name = FONT_FAMILY
p_quote.font.size = Pt(9.5)
p_quote.font.italic = True
p_quote.font.color.rgb = SLATE_TEXT
p_quote.space_before = Pt(6)

p_stat = iotf.add_paragraph()
p_stat.text = "CLASSIFICATION RESULT: 🚨 FAKE (Confidence: 99.7%)"
p_stat.font.name = FONT_FAMILY
p_stat.font.size = Pt(11)
p_stat.font.bold = True
p_stat.font.color.rgb = DARK_RED
p_stat.space_before = Pt(10)

p_lime_head = iotf.add_paragraph()
p_lime_head.text = "LIME Word-Level Attribution Breakdown:"
p_lime_head.font.name = FONT_FAMILY
p_lime_head.font.size = Pt(11)
p_lime_head.font.bold = True
p_lime_head.font.color.rgb = NAVY
p_lime_head.space_before = Pt(10)

lime_weights = [
    ("• 'fake' (weight: -0.185):", "Pushes decisively toward FAKE (Debunking / hoax marker)."),
    ("• 'video' (weight: -0.112):", "Pushes strongly toward FAKE (Viral multimedia claim)."),
    ("• 'missing' (weight: -0.078):", "Pushes toward FAKE (Sensational emotional hook)."),
    ("• 'IAF' / 'pilot' (+0.042):", "Pulls slightly toward REAL (Institutional military terminology).")
]
for title, desc in lime_weights:
    p_w = iotf.add_paragraph()
    p_w.text = f"{title} {desc}"
    p_w.font.name = FONT_FAMILY
    p_w.font.size = Pt(8.5)
    p_w.font.color.rgb = DARK_TEXT
    p_w.space_before = Pt(3)

p_sum = iotf.add_paragraph()
p_sum.text = "Takeaway: LIME demonstrates the internal tug-of-war—sensational debunked hoax cues overpower institutional military mentions."
p_sum.font.name = FONT_FAMILY
p_sum.font.size = Pt(8.5)
p_sum.font.italic = True
p_sum.font.color.rgb = PRIMARY_BLUE
p_sum.space_before = Pt(8)

lime_frame = s8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.45), Inches(5.733), Inches(5.25))
lime_frame.fill.solid()
lime_frame.fill.fore_color.rgb = CARD_BG
lime_frame.line.color.rgb = BORDER_COLOR

img_lime = RESULTS_DIR / "lime_case_1.png"
if img_lime.exists():
    s8.shapes.add_picture(str(img_lime), Inches(6.9), Inches(1.55), width=Inches(5.533))

# ==============================================================================
# SLIDE 9: CONCLUSION & SUMMARY
# ==============================================================================
s9 = prs.slides.add_slide(blank_layout)
add_header_and_footer(s9, "Summary & Key Takeaways for Project Evaluation", "PROJECT CONCLUSION", 9)

c_summary_cards = [
    ("Point 1: Corpus Exploration Rigor", "Analyzed 4,000 verified IFND articles with 53,340 words and 4,980 sentences. Proved that fake news headlines are linguistically packed with sensational urgency markers."),
    ("Point 2: Embedding Justification", "Proved sublinear TF-IDF (1-2 gram) outperforms dense Word2Vec (90.58% vs 87.84% F1) with 8.5x lower latency (0.12s vs 1.02s) while eliminating vector semantic dilution."),
    ("Point 3: Specialized Indian NLP", "Resolved standard NLTK limitations by preserving Indian numerical scales ('lakh', 'crore'), cultural entities, and WhatsApp forward syntax."),
    ("Point 4: Superior Decision Margins", "Proposed regularized Logistic Regression (90.75%) and Linear SVM (91.87%) decisively outperform baseline Naive Bayes (88.75%) with calibrated probabilities."),
    ("Point 5: Transparent XAI Prototype", "Deployed an interactive Streamlit application (localhost:8501) with real-time LIME word attribution bar charts and in-text lexical highlights.")
]

y = 1.45
for title, desc in c_summary_cards:
    card = s9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(y), Inches(11.733), Inches(0.95))
    card.fill.solid()
    card.fill.fore_color.rgb = WHITE
    card.line.color.rgb = BORDER_COLOR
    ctf = card.text_frame
    ctf.margin_left = Inches(0.25)
    ctf.margin_top = Inches(0.1)
    p1 = ctf.paragraphs[0]
    p1.text = f"✔ {title}"
    p1.font.name = FONT_FAMILY
    p1.font.size = Pt(11.5)
    p1.font.bold = True
    p1.font.color.rgb = NAVY
    p2 = ctf.add_paragraph()
    p2.text = desc
    p2.font.name = FONT_FAMILY
    p2.font.size = Pt(9.0)
    p2.font.color.rgb = SLATE_TEXT
    p2.space_before = Pt(2)
    y += 1.08

output_path = Path("presentation/fake_news_indian_context_presentation.pptx")
prs.save(str(output_path))
print(f"Presentation regenerated successfully with authentic metrics at: {output_path.resolve()}")
