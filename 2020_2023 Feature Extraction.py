import pandas as pd
import spacy
import re
import textstat
from collections import Counter
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import textstat
from spacy.lang.en.stop_words import STOP_WORDS
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# spaCy model
nlp = spacy.load("en_core_web_sm")
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
vader = SentimentIntensityAnalyzer()

# Data Loading
df_2020 = pd.read_csv("2020_data_clean_final.csv")
df_2021 = pd.read_csv("2021_Final_data_clean_test.csv")
df_2022 = pd.read_csv("2022_Final_data_clean_test.csv")
df_2023 = pd.read_csv("2023_data_clean_test.csv")

combined_df = pd.concat([df_2020, df_2021, df_2022,df_2023], ignore_index=True)
combined_df.head(100)

# Reorder to: forum, year, decision, avg_score, INTRODUCTION, CONCLUSION
ordered_cols = ['forum', 'year', 'decision', 'avg_score', 'INTRODUCTION', 'CONCLUSION']
# Only keep those that exist in the current DataFrame
existing_cols = [col for col in ordered_cols if col in combined_df.columns]

filtered_df = combined_df[existing_cols]
filtered_df.head(100)
filtered_df.to_csv("filtered.csv")

# ======= Feature Extraction =================
# Preload hype_list
hype_dict = {
    'Importance': ['compelling', 'critical', 'crucial', 'essential', 'foundational', 'fundamental', 'imperative', 'important', 'indispensable', 'invaluable', 'key', 'major', 'paramount', 'pivotal', 'significant', 'strategic', 'timely', 'ultimate', 'urgent', 'vital'],
    'Novelty': ['creative', 'emerging', 'first', 'groundbreaking', 'innovative', 'latest', 'novel', 'revolutionary', 'unique', 'unparalleled', 'unprecedented'],
    'Rigor': ['accurate', 'advanced', 'careful', 'cohesive', 'detailed', 'nuanced', 'powerful', 'quality', 'reproducible', 'rigorous', 'robust', 'scientific', 'sophisticated', 'strong', 'systematic'],
    'Scale': ['ample', 'biggest', 'broad', 'comprehensive', 'considerable', 'deeper', 'diverse', 'enormous', 'expansive', 'extensive', 'fastest', 'greatest', 'huge', 'immediate', 'immense', 'interdisciplinary', 'international', 'interprofessional', 'largest', 'massive', 'multidisciplinary', 'myriad', 'overwhelming', 'substantial', 'top', 'transdisciplinary', 'tremendous', 'vast'],
    'Utility': ['accessible', 'actionable', 'deployable', 'durable', 'easy', 'effective', 'efficacious', 'efficient', 'generalizable', 'ideal', 'impactful', 'intuitive', 'meaningful', 'productive', 'ready', 'relevant', 'rich', 'safer', 'scalable', 'seamless', 'sustainable', 'synergistic', 'tailored', 'tangible', 'transformative', 'userfriendly'],
    'Quality': ['ambitious', 'collegial', 'dedicated', 'exceptional', 'experienced', 'intellectual', 'longstanding', 'motivated', 'premier', 'prestigious', 'promising', 'qualified', 'renowned', 'senior', 'skilled', 'stellar', 'successful', 'talented', 'vibrant'],
    'Attitude': ['attractive', 'confident', 'exciting', 'incredible', 'interesting', 'intriguing', 'notable', 'outstanding', 'remarkable', 'surprising'],
    'Problem': ['alarming', 'daunting', 'desperate', 'devastating', 'dire', 'dismal', 'elusive', 'stark', 'unanswered', 'unmet'],
}
hype_list = list(set([word for words in hype_dict.values() for word in words]))

#Hedge words
hedge_words = {
    "can", "could", "may", "might", "should", "would",
    "advise", "advocate", "agree with", "allege", "anticipate", "appear",
    "argue", "assert", "assume", "attempt", "believe", "calculate",
    "conjecture", "contend", "consider", "demonstrate", "display", "doubt",
    "estimate", "expect", "feel", "find", "guess", "hint", "hope",
    "hypothesize", "implicate", "imply", "indicate", "insinuate", "intend",
    "intimate", "maintain", "mention", "observe", "offer", "opine",
    "postulate", "predict", "presume", "prone to", "propose", "proposition",
    "reckon", "recommend", "report", "reveal", "seem", "show", "signal",
    "speculate", "suggest", "support", "suppose", "surmise", "suspect",
    "tend to", "think", "try to", "subtle", "suggested", "in tune with",
    "uncertain", "unlikely", "in line with", "potential", "mainly",
    "mildly", "moderately", "mostly", "near", "nearly", "not always",
    "occasionally", "often", "partially", "partly", "passably", "much",
    "not all", "on occasion", "several", "perhaps", "possibly",
    "potentially", "predictably", "presumably", "primarily", "probably",
    "quite", "rarely", "rather", "reasonably", "relatively", "roughly",
    "about", "admittedly", "all but", "almost", "approximately", "arguably",
    "around", "averagely", "fairly", "frequently", "generally", "hardly",
    "largely", "a few", "few", "little", "more or less", "most", "scarcely",
    "seemingly", "slightly", "sometimes", "somewhat", "subtly", "supposedly",
    "tolerably", "usually", "virtually", "to a lesser extent",
    "to a minor extent", "to an extent", "to some extent",
    "agreement with", "assertion", "assumption", "attempt", "belief",
    "chance", "claim", "expectation", "guidance", "implication", "intention",
    "majority", "possibility", "prediction", "presupposition", "probability",
    "proposal", "recommendation", "suggestion", "tendency"
}

# === Core Feature Functions ===
def extract_features(text):
    """Extract clause and structure counts from text."""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return dict.fromkeys([
            "nominalization_count", "passive_voice_count", "noun_chain_count", "colon_count",
            "finite_complement_clause_count", "nonfinite_complement_clause_count",
            "finite_adverbial_clause_count", "finite_relative_clause_count",
            "nonfinite_relative_clause_count"
        ], 0)

    features = {key: 0 for key in [
        "nominalization_count", "passive_voice_count", "noun_chain_count", "colon_count",
        "finite_complement_clause_count", "nonfinite_complement_clause_count",
        "finite_adverbial_clause_count", "finite_relative_clause_count",
        "nonfinite_relative_clause_count"]}
    features["colon_count"] = text.count(':')

    doc = nlp(text)

    # Passive voice
    features["passive_voice_count"] = sum(1 for token in doc if token.dep_ == "nsubjpass")

    # Nominalization
    nominal_suffixes = ["tion", "sion", "ment", "ness", "ity", "ence", "ance", "ship"]
    features["nominalization_count"] = sum(
        1 for token in doc if token.pos_ == "NOUN" and any(token.text.lower().endswith(suf) for suf in nominal_suffixes))

    # Noun chains
    noun_chain_len = 0
    for token in doc:
        if token.pos_ == "NOUN":
            noun_chain_len += 1
        else:
            if noun_chain_len >= 2:
                features["noun_chain_count"] += 1
            noun_chain_len = 0

    # Clause patterns
    for sent in doc.sents:
        s = sent.text.lower()
        if re.search(r'\b(that|what|whether|if)\b', s):
            features["finite_complement_clause_count"] += 1
        if re.search(r'\bto\\s+\\w+|\\w+ing\b', s):
            features["nonfinite_complement_clause_count"] += 1
        if re.search(r'\b(because|although|while|when|since|if)\b', s):
            features["finite_adverbial_clause_count"] += 1
        if re.search(r'\b(that|which|who|whose|whom)\b', s):
            features["finite_relative_clause_count"] += 1

    return features

def extract_writing_style_features(text):
    """Compute paragraph, lexical, POS, bigram, readability, sentiment, and syntactic metrics."""
    features = {}
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    para_lengths = [len(p.split()) for p in paragraphs]
    features['avg_paragraph_length'] = sum(para_lengths) / len(para_lengths) if para_lengths else 0

    doc = nlp(text)
    words = [token.text.lower() for token in doc if token.is_alpha]
    sents = list(doc.sents)

    features.update({
        'num_words': len(words),
        'num_unique_words': len(set(words)),
        'num_sentences': len(sents),
        'avg_sentence_length': len(words) / len(sents) if sents else 0,
    })

    pos_counts = Counter([token.pos_ for token in doc])
    total_tokens = len(doc)
    for pos in ['NOUN', 'VERB', 'ADJ', 'ADV']:
        features[f'{pos.lower()}_ratio'] = pos_counts.get(pos, 0) / total_tokens if total_tokens else 0

    verbs = [token for token in doc if token.pos_ == 'VERB']
    present = sum(1 for v in verbs if v.tag_ in ['VBP', 'VBZ'])
    past = sum(1 for v in verbs if v.tag_ == 'VBD')
    perfect = sum(1 for v in verbs if v.tag_ == 'VBN' and any(c.tag_ in ['VBZ', 'HVB'] for c in v.children))
    passive = sum(1 for v in verbs if v.tag_ == 'VBN' and any(c.dep_ == 'auxpass' for c in v.children))
    features.update({
        'num_verbs': len(verbs),
        'verb_type_count': len(set([v.lemma_ for v in verbs])),
        'present_ratio': present / len(verbs) if verbs else 0,
        'past_ratio': past / len(verbs) if verbs else 0,
        'perfect_ratio': perfect / len(verbs) if verbs else 0,
        'passive_voice_ratio': passive / len(verbs) if verbs else 0
    })

    all_words_nostop = [token.text.lower() for token in doc if token.is_alpha and token.text.lower() not in STOP_WORDS]
    bigrams = zip(all_words_nostop[:-1], all_words_nostop[1:])
    cleaned_bigram_counter = Counter([" ".join(b) for b in bigrams if 'et' not in b])
    features['cleaned_bigram_total_count'] = sum(cleaned_bigram_counter.values())

    features['ref_count'] = len(re.findall(r'\(([^)]*?\\d{4})\)', text))
    features['promo_word_ratio'] = sum(word in hype_list for word in words) / len(words) if words else 0
    features['hedge_word_ratio'] = sum(word in hedge_words for word in words) / len(words) if words else 0

    blob = TextBlob(text)
    features['polarity'] = blob.sentiment.polarity
    features['subjectivity'] = blob.sentiment.subjectivity

    features['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
    features['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text)
    features['automated_readability_index'] = textstat.automated_readability_index(text)
    features['dale_chall_readability_score'] = textstat.dale_chall_readability_score(text)

    features.update(extract_syntactic_complexity(doc))
    return features


def extract_syntactic_complexity(doc):
    """Calculate classic syntactic complexity metrics."""
    counts = Counter({
        'words': sum(1 for token in doc if token.is_alpha),
        'sentences': len(list(doc.sents)),
        'clauses': 0, 't_units': 0, 'complex_t_units': 0,
        'dependent_clauses': 0, 'coordinate_phrases': 0,
        'complex_nominals': 0, 'verb_phrases': 0
    })

    for sent in doc.sents:
        root = sent.root
        if root.dep_ in ('ROOT', 'ccomp', 'advcl', 'xcomp'):
            counts['t_units'] += 1
            if any(c.dep_ in ('advcl', 'ccomp', 'xcomp', 'relcl') for c in root.children):
                counts['complex_t_units'] += 1
        for token in sent:
            if token.dep_ in ('ccomp', 'advcl', 'relcl', 'xcomp'):
                counts['clauses'] += 1
                counts['dependent_clauses'] += 1
            if token.dep_ == 'conj':
                counts['coordinate_phrases'] += 1
            if token.dep_ in ('attr', 'appos', 'nmod'):
                counts['complex_nominals'] += 1
            if token.pos_ == 'VERB' and token.dep_ not in ('aux', 'auxpass'):
                counts['verb_phrases'] += 1
    counts['clauses'] += counts['sentences']

    return {
        'MLC': counts['words'] / counts['clauses'] if counts['clauses'] else 0,
        'MLS': counts['words'] / counts['sentences'] if counts['sentences'] else 0,
        'MLT': counts['words'] / counts['t_units'] if counts['t_units'] else 0,
        'C_S': counts['clauses'] / counts['sentences'] if counts['sentences'] else 0,
        'C_T': counts['clauses'] / counts['t_units'] if counts['t_units'] else 0,
        'CT_T': counts['complex_t_units'] / counts['t_units'] if counts['t_units'] else 0,
        'DC_C': counts['dependent_clauses'] / counts['clauses'] if counts['clauses'] else 0,
        'DC_T': counts['dependent_clauses'] / counts['t_units'] if counts['t_units'] else 0,
        'CP_C': counts['coordinate_phrases'] / counts['clauses'] if counts['clauses'] else 0,
        'CP_T': counts['coordinate_phrases'] / counts['t_units'] if counts['t_units'] else 0,
        'T_S': counts['t_units'] / counts['sentences'] if counts['sentences'] else 0,
        'CN_C': counts['complex_nominals'] / counts['clauses'] if counts['clauses'] else 0,
        'CN_T': counts['complex_nominals'] / counts['t_units'] if counts['t_units'] else 0,
        'VP_T': counts['verb_phrases'] / counts['t_units'] if counts['t_units'] else 0
    }

# === Sentence-level Structural Features ===
# Based on discourse structure and lexical signals
# Source: https://aclanthology.org/N19-1361.pdf
def extract_structural_features(sentence, sentence_index, total_sentences):
    features = {}
    transition_words = [
        'however', 'although', 'whereas', 'nonetheless', 'moreover', 'furthermore', 'nevertheless',
        'therefore', 'thus', 'despite', 'yet', 'consequently', 'alternatively', 'in contrast',
        'on the other hand', 'similarly', 'likewise', 'regardless', 'even though', 'notwithstanding'
    ]
    features['sentence_position_ratio'] = sentence_index / total_sentences if total_sentences else 0  # Normalized position
    features['sentence_length_characters'] = len(sentence)  # Sentence length in characters
    features['has_transition_word'] = int(any(tw in sentence.lower() for tw in transition_words))  # Discourse marker presence
    features['num_citations_in_sentence'] = len(re.findall(r'\(([^)]*?\d{4})\)', sentence))  # Citations (e.g., APA format)
    return features


# === Topic Similarity Features ===
def extract_topic_similarity_features(sentence, intro_emb, conclusion_emb):
    """
    Compute sentence-to-introduction and sentence-to-conclusion cosine similarity
    using MiniLM sentence embeddings.
    """
    features = {}
    sent_emb = sentence_model.encode([sentence])[0].reshape(1, -1)
    features['topic_similarity_to_intro'] = cosine_similarity(sent_emb, intro_emb.reshape(1, -1))[0][0]
    features['topic_similarity_to_conclusion'] = cosine_similarity(sent_emb, conclusion_emb.reshape(1, -1))[0][0]
    return features

# === Apply All Feature Modules to DataFrame ===
feature_list = []

# Precompute embeddings for full sections
intro_embeddings = sentence_model.encode(filtered_df['INTRODUCTION'].fillna("").tolist())
conclusion_embeddings = sentence_model.encode(filtered_df['CONCLUSION'].fillna("").tolist())

for i, row in tqdm(filtered_df.iterrows(), total=len(filtered_df)):
    combined_features = {'index': i}
    intro_emb = intro_embeddings[i].reshape(1, -1)
    conclusion_emb = conclusion_embeddings[i].reshape(1, -1)

    for section_name in ['INTRODUCTION', 'CONCLUSION']:
        text = str(row[section_name]) if pd.notnull(row[section_name]) else ""

        # Section-level writing features
        para_features = extract_writing_style_features(text)
        para_features = {f"{section_name}_{k}": v for k, v in para_features.items()}
        combined_features.update(para_features)

        # Sentence-level structure + topic features
        doc = nlp(text)
        total_sentences = len(list(doc.sents))
        struct_topic_features_list = []

        for sent_idx, sent in enumerate(doc.sents):
            sentence_text = sent.text
            struct_feats = extract_structural_features(sentence_text, sent_idx, total_sentences)
            topic_feats = extract_topic_similarity_features(sentence_text, intro_emb, conclusion_emb)
            struct_topic_features_list.append({**struct_feats, **topic_feats})

        # Aggregate sentence-level features (mean, max, std)
        if struct_topic_features_list:
            df_sent_feats = pd.DataFrame(struct_topic_features_list)
            for col in df_sent_feats.columns:
                combined_features[f'{section_name}_{col}_mean'] = df_sent_feats[col].mean()
                combined_features[f'{section_name}_{col}_max'] = df_sent_feats[col].max()
                combined_features[f'{section_name}_{col}_std'] = df_sent_feats[col].std()

    feature_list.append(combined_features)

# Assemble final feature dataframe
feature_df = pd.DataFrame(feature_list).set_index('index')
df_with_features = pd.concat([filtered_df, feature_df], axis=1).fillna(0)






