import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
import io
import base64
from io import BytesIO
warnings.filterwarnings('ignore')

# Set matplotlib backend dan style
plt.switch_backend('Agg')  # Non-interactive backend
plt.style.use('default')
sns.set_palette("husl")

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

# Visualization Libraries  
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================================
# TEXT PROCESSING SETUP WITH ROBUST ERROR HANDLING
# ============================================================================

def setup_text_processing():
    """Setup text processing libraries dengan error handling"""
    global NLTK_AVAILABLE, SASTRAWI_AVAILABLE, stemmer, stopword_remover
    
    # Initialize NLTK
    NLTK_AVAILABLE = False
    try:
        import nltk
        from nltk.tokenize import word_tokenize
        
        # Test basic tokenization
        word_tokenize("test")
        NLTK_AVAILABLE = True
        
    except Exception as e:
        NLTK_AVAILABLE = False
    
    # Initialize Sastrawi
    SASTRAWI_AVAILABLE = False
    stemmer = None
    stopword_remover = None
    
    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
        
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        
        stopword_factory = StopWordRemoverFactory()
        stopword_remover = stopword_factory.create_stop_word_remover()
        
        # Test stemmer
        test_result = stemmer.stem("testing")
        SASTRAWI_AVAILABLE = True
        
    except Exception as e:
        SASTRAWI_AVAILABLE = False
    
    return NLTK_AVAILABLE, SASTRAWI_AVAILABLE

# Initialize text processing
NLTK_AVAILABLE, SASTRAWI_AVAILABLE = setup_text_processing()

# ============================================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Sentiment Analysis - Naive Bayes",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def safe_division(numerator, denominator, default=0.0):
    """Safe division dengan handling zero"""
    try:
        if denominator == 0:
            return default
        return float(numerator) / float(denominator)
    except:
        return default

def ensure_numeric(value, default=0.0):
    """Ensure value is numeric"""
    try:
        return float(value)
    except:
        return default

def clear_matplotlib():
    """Clear matplotlib figures properly"""
    try:
        plt.clf()
        plt.close('all')
    except:
        pass

# ============================================================================
# TEXT PREPROCESSING FUNCTIONS
# ============================================================================

def clean_text(text):
    """Membersihkan text dari karakter tidak diinginkan"""
    if pd.isna(text) or text is None:
        return ""
    
    try:
        text = str(text)
        
        # Hapus URL, mention, hashtag
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Hapus emoji dan karakter khusus
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        
        # Convert ke lowercase
        text = text.lower()
        
        # Hapus extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    except:
        return ""

def simple_tokenize(text):
    """Simple tokenization yang aman"""
    if not text:
        return []
    try:
        tokens = [token.strip() for token in str(text).split() if token.strip() and len(token.strip()) > 1]
        return tokens
    except:
        return []

def safe_word_tokenize(text):
    """Safe tokenization dengan fallback"""
    if not text:
        return []
    
    try:
        if NLTK_AVAILABLE:
            import nltk
            from nltk.tokenize import word_tokenize
            return word_tokenize(str(text))
        else:
            return simple_tokenize(text)
    except:
        return simple_tokenize(text)

def preprocess_text(text):
    """Preprocessing text yang robust"""
    if not text or pd.isna(text):
        return ""
    
    try:
        # Clean text
        text = clean_text(text)
        if not text:
            return ""
        
        # Tokenisasi
        tokens = safe_word_tokenize(text)
        if not tokens:
            return ""
        
        # Filter token yang terlalu pendek (kecuali kata penting)
        important_short_words = {'ok', 'ga', 'gak','ya','no'}
        tokens = [token for token in tokens if len(token) > 2 or token in important_short_words]
        
        # ✅ FIXED: Enhanced stopword removal - TIDAK hapus kata sentiment
        basic_stopwords = {
            'yang', 'dan', 'di', 'ke', 'dari', 'untuk', 'dengan', 'pada', 'adalah', 'ini', 'itu',
            'atau', 'juga', 'akan', 'ada', 'dalam', 'oleh', 'dapat', 'bisa', 'sudah', 'telah',
            'saya', 'kamu', 'kami', 'mereka', 'nya', 'ter', 'se'
            # ✅ PENTING: JANGAN masukkan 'tidak', 'suka', 'bagus', 'buruk' di sini!
        }
        
        # ✅ FIXED: Daftar kata sentiment yang HARUS dipertahankan
        sentiment_keywords = {
            'tidak', 'bukan', 'jangan', 'kurang', 'suka', 'senang', 'kecewa', 
            'bagus', 'jelek', 'enak', 'buruk', 'mantap', 'zonk', 'recommended',
            'mengecewakan', 'puas', 'excellent', 'terbaik', 'nyaman', 'ramah',
            'berkualitas', 'amazing', 'perfect', 'pahit', 'mahal', 'lambat',
            'kotor', 'berisik', 'terrible', 'bad', 'good', 'great'
        }
        
        # ✅ FIXED: Filter dengan logika yang benar
        filtered_tokens = []
        for token in tokens:
            # Pertahankan jika: bukan stopword ATAU mengandung sentiment
            if token not in basic_stopwords or any(sentiment in token for sentiment in sentiment_keywords):
                filtered_tokens.append(token)
        
        return ' '.join(filtered_tokens) if filtered_tokens else ""
    
    except Exception as e:
        return ""

def create_sentiment_label(rating):
    """Convert rating menjadi sentiment label"""
    try:
        rating = ensure_numeric(rating, 30)
        return 'Negatif' if rating <= 30 else 'Positif'
    except:
        return 'Positif'

# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

@st.cache_data
def generate_sample_data():
    """Generate sample data yang lebih balanced dan realistic"""
    np.random.seed(42)
    
    # Kata-kata yang lebih distingtif untuk setiap sentiment
    positive_words = [
        'enak', 'mantap', 'bagus', 'recommended', 'puas', 'suka', 'excellent', 
        'terbaik', 'nyaman', 'ramah', 'berkualitas', 'amazing', 'perfect',
        'lezat', 'keren', 'oke', 'top', 'juara', 'favorit', 'istimewa'
    ]
    
    negative_words = [
        'buruk', 'jelek', 'mengecewakan', 'tidak suka', 'zonk', 'pahit', 
        'mahal', 'lambat', 'kotor', 'berisik', 'terrible', 'bad', 'awful',
        'mengerikan', 'payah', 'hancur', 'rusak', 'benci', 'kecewa'
    ]
    
    neutral_words = [
        'tempat', 'kopi', 'cafe', 'resto', 'makanan', 'minuman', 'pelayanan', 'staff', 'harga', 'menu', 'suasana', 'lokasi','the', 'she', 'when',
    ]
    
    
    def generate_review(sentiment_type, length=10):
        """Generate review dengan sentiment yang jelas"""
        words = []
        
        if sentiment_type == 'positive':
            # 60% positive words, 30% neutral, 10% connector
            for _ in range(length):
                rand = np.random.random()
                if rand < 0.6:
                    words.append(np.random.choice(positive_words))
                elif rand < 0.9:
                    words.append(np.random.choice(neutral_words))
                else:
                    words.append(np.random.choice(['sangat', 'banget', 'sekali', 'dan', 'dengan']))
        else:
            # 60% negative words, 30% neutral, 10% connector
            for _ in range(length):
                rand = np.random.random()
                if rand < 0.6:
                    words.append(np.random.choice(negative_words))
                elif rand < 0.9:
                    words.append(np.random.choice(neutral_words))
                else:
                    words.append(np.random.choice(['sangat', 'banget', 'sekali', 'dan', 'dengan']))
        
        return ' '.join(words)
    
    data = []
    
    # Generate 50% positive (rating 40-50)
    for i in range(150):  # 50% dari 357
        rating = np.random.choice([40, 50], p=[0.3, 0.7])
        review = generate_review('positive', np.random.randint(8, 15))
        data.append({
            'page': 1,
            'user_name': f'User_{i+1}',
            'rating': rating,
            'review_text': review,
            'image_count': np.random.randint(0, 10)
        })
    
    # Generate 50% negative (rating 10-30)
    for i in range(150):  # 50% dari 357
        rating = np.random.choice([10, 20, 30], p=[0.4, 0.3, 0.3])
        review = generate_review('negative', np.random.randint(8, 15))
        data.append({
            'page': 214 + i,
            'user_name': f'User_{214 + i + 1}',
            'rating': rating,
            'review_text': review,
            'image_count': np.random.randint(0, 10)
        })
    
    # Shuffle data
    df = pd.DataFrame(data)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

def load_uploaded_data(uploaded_file):
    """Load data dari file upload dengan error handling"""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            # Try different encodings and separators
            content = uploaded_file.read()
            
            # Try UTF-8 first
            try:
                df = pd.read_csv(io.StringIO(content.decode('utf-8')))
            except UnicodeDecodeError:
                df = pd.read_csv(io.StringIO(content.decode('latin-1')))
            
            # If only one column, try semicolon
            if len(df.columns) == 1:
                uploaded_file.seek(0)
                content = uploaded_file.read()
                try:
                    df = pd.read_csv(io.StringIO(content.decode('utf-8')), sep=';')
                except UnicodeDecodeError:
                    df = pd.read_csv(io.StringIO(content.decode('latin-1')), sep=';')
        
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Format file tidak didukung. Gunakan CSV atau Excel.")
            return None
        
        return df
    
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def process_data(df):
    """Process dan clean data dengan robust error handling"""
    try:
        # Check required columns
        required_cols = ['review_text', 'rating']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"Kolom yang dibutuhkan tidak ditemukan: {missing_cols}")
            return None
        
        # Data cleaning
        df_clean = df.copy()
        
        # Handle missing values
        df_clean = df_clean.dropna(subset=['review_text'])
        df_clean = df_clean[df_clean['review_text'].astype(str).str.strip() != '']
        
        if len(df_clean) == 0:
            st.error("Tidak ada data valid setelah cleaning")
            return None
        
        # Create sentiment labels
        df_clean['sentiment'] = df_clean['rating'].apply(create_sentiment_label)
        
        # Text preprocessing dengan progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        processed_texts = []
        for i, text in enumerate(df_clean['review_text']):
            processed_text = preprocess_text(text)
            processed_texts.append(processed_text)
            
            # Update progress
            progress = (i + 1) / len(df_clean)
            progress_bar.progress(progress)
            status_text.text(f'Processing text {i+1}/{len(df_clean)}')
        
        df_clean['review_text_clean'] = processed_texts
        
        # Remove empty texts after preprocessing
        df_clean = df_clean[df_clean['review_text_clean'].astype(str).str.strip() != '']
        
        progress_bar.empty()
        status_text.empty()
        
        if len(df_clean) == 0:
            st.error("Tidak ada data valid setelah preprocessing")
            return None
        
        return df_clean
    
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return None

# ============================================================================
# MACHINE LEARNING FUNCTIONS
# ============================================================================

def safe_metrics_calculation(y_true, y_pred, pos_label='Positif'):
    """Safe metrics calculation dengan error handling"""
    try:
        # Ensure we have valid data
        if len(y_true) == 0 or len(y_pred) == 0:
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)
        
        # Check if we have both classes
        unique_true = set(y_true)
        unique_pred = set(y_pred)
        
        if len(unique_true) == 1 or pos_label not in unique_true:
            # Single class scenario
            precision = 1.0 if pos_label in unique_pred else 0.0
            recall = 1.0 if pos_label in unique_true else 0.0
            f1 = safe_division(2 * precision * recall, precision + recall, 0.0)
        else:
            # Multi-class scenario
            precision = precision_score(y_true, y_pred, pos_label=pos_label, average='binary', zero_division=0.0)
            recall = recall_score(y_true, y_pred, pos_label=pos_label, average='binary', zero_division=0.0)
            f1 = f1_score(y_true, y_pred, pos_label=pos_label, average='binary', zero_division=0.0)
        
        return {
            'accuracy': ensure_numeric(accuracy, 0.0),
            'precision': ensure_numeric(precision, 0.0),
            'recall': ensure_numeric(recall, 0.0),
            'f1': ensure_numeric(f1, 0.0)
        }
    
    except Exception as e:
        st.warning(f"Error calculating metrics: {str(e)}")
        return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

@st.cache_data
def train_models(_df_clean):
    """Train Naive Bayes models dengan robust error handling dan debugging"""
    try:
        if len(_df_clean) < 5:
            st.error("Dataset terlalu kecil untuk training (minimal 5 samples)")
            return None
        
        # Prepare data
        X_text = _df_clean['review_text_clean']
        y = _df_clean['sentiment']
        
        # Debug: Show data distribution
        st.write("**Training Debug Info:**")
        class_counts = y.value_counts()
        st.write(f"Class distribution: {dict(class_counts)}")
        
        if len(class_counts) < 2:
            st.error("Dataset harus memiliki minimal 2 kelas sentiment")
            return None
        
        min_class_count = class_counts.min()
        if min_class_count < 2:
            st.warning(f"Kelas minoritas hanya memiliki {min_class_count} sample")
        
        # Show sample processed texts for each class
        st.write("**Sample processed texts:**")
        for sentiment in y.unique():
            sample_texts = X_text[y == sentiment].head(3).tolist()
            st.write(f"{sentiment}: {sample_texts}")
        
        # Vectorization dengan parameter yang lebih baik
        try:
            # Bag of Words - lebih conservative
            bow_vectorizer = CountVectorizer(
                max_features=500,  # Reduced features
                min_df=1,
                max_df=0.95,  # Lebih inclusive
                ngram_range=(1, 2),  # Hanya unigram
                lowercase=True,
                strip_accents='unicode',
                token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b'
            )
            X_bow = bow_vectorizer.fit_transform(X_text)
            
            # Debug vectorizer
            st.write(f"BoW vocabulary size: {len(bow_vectorizer.vocabulary_)}")
            st.write(f"BoW matrix shape: {X_bow.shape}")
            
            # TF-IDF
            tfidf_vectorizer = TfidfVectorizer(
                max_features=500,
                min_df=1,
                max_df=0.95,
                ngram_range=(1, 2),
                lowercase=True,
                strip_accents='unicode',
                token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b',
                use_idf=True,
                smooth_idf=True,
                sublinear_tf=True
            )
            X_tfidf = tfidf_vectorizer.fit_transform(X_text)
            
            st.write(f"TF-IDF vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")
            st.write(f"TF-IDF matrix shape: {X_tfidf.shape}")
            
            # Show some feature names
            feature_names = bow_vectorizer.get_feature_names_out()
            st.write(f"Sample features: {list(feature_names[:20])}")
            
        except Exception as e:
            st.error(f"Error in vectorization: {str(e)}")
            return None
        
        # Split data dengan stratification
        test_size = min(0.3, max(0.2, min_class_count * 0.4 / len(X_text)))
        st.write(f"Test size: {test_size:.2f}")
        
        try:
            X_bow_train, X_bow_test, y_train, y_test = train_test_split(
                X_bow, y, test_size=test_size, random_state=42, stratify=y
            )
            X_tfidf_train, X_tfidf_test, _, _ = train_test_split(
                X_tfidf, y, test_size=test_size, random_state=42, stratify=y
            )
            
            st.write(f"Training set distribution: {dict(y_train.value_counts())}")
            st.write(f"Test set distribution: {dict(y_test.value_counts())}")
            
        except ValueError as e:
            st.warning(f"Stratification failed: {e}. Using random split.")
            X_bow_train, X_bow_test, y_train, y_test = train_test_split(
                X_bow, y, test_size=test_size, random_state=42
            )
            X_tfidf_train, X_tfidf_test, _, _ = train_test_split(
                X_tfidf, y, test_size=test_size, random_state=42
            )
        
        # Train models
        try:
            # Naive Bayes + BoW dengan alpha yang lebih besar untuk smoothing
            nb_bow = MultinomialNB(alpha=0.1)
            nb_bow.fit(X_bow_train, y_train)
            y_pred_bow = nb_bow.predict(X_bow_test)
            
            # Debug model
            st.write(f"BoW model classes: {nb_bow.classes_}")
            st.write(f"BoW class log priors: {nb_bow.class_log_prior_}")
            
            # Naive Bayes + TF-IDF
            nb_tfidf = MultinomialNB(alpha=0.1)
            nb_tfidf.fit(X_tfidf_train, y_train)
            y_pred_tfidf = nb_tfidf.predict(X_tfidf_test)
            
            st.write(f"TF-IDF model classes: {nb_tfidf.classes_}")
            st.write(f"TF-IDF class log priors: {nb_tfidf.class_log_prior_}")
            
        except Exception as e:
            st.error(f"Error training models: {str(e)}")
            return None
        
        # Calculate metrics safely
        metrics_bow = safe_metrics_calculation(y_test, y_pred_bow)
        metrics_tfidf = safe_metrics_calculation(y_test, y_pred_tfidf)
        
        st.write(f"BoW metrics: {metrics_bow}")
        st.write(f"TF-IDF metrics: {metrics_tfidf}")
        
        return {
            'models': {'bow': nb_bow, 'tfidf': nb_tfidf},
            'vectorizers': {'bow': bow_vectorizer, 'tfidf': tfidf_vectorizer},
            'metrics': {'bow': metrics_bow, 'tfidf': metrics_tfidf},
            'predictions': {'bow': y_pred_bow, 'tfidf': y_pred_tfidf},
            'test_data': {'y_test': y_test},
            'train_info': {
                'total_samples': len(_df_clean),
                'train_samples': len(y_train),
                'test_samples': len(y_test),
                'classes': list(y.unique()),
                'class_distribution': dict(y.value_counts()),
                'feature_names': {
                    'bow': feature_names[:50].tolist(),  # Store some feature names for debugging
                    'tfidf': tfidf_vectorizer.get_feature_names_out()[:50].tolist()
                }
            }
        }
    
    except Exception as e:
        st.error(f"Unexpected error in model training: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None

def predict_sentiment(text, model_type='tfidf'):
    """Prediksi sentiment untuk text baru dengan debugging"""
    try:
        if 'results' not in st.session_state or st.session_state.results is None:
            st.error("Model tidak tersedia")
            return None, None
        
        # Check if debug should be shown
        show_debug = st.session_state.get('show_debug', False)
        
        if show_debug:
            # Debug: Show original text
            st.write("**Debug Info:**")
            st.write(f"Original text: {text[:100]}...")
        
        # Preprocess text
        clean_text = preprocess_text(text)
        
        if show_debug:
            st.write(f"Preprocessed text: {clean_text[:100]}...")
        
        if not clean_text or clean_text.strip() == "":
            if show_debug:
                st.warning("Text kosong setelah preprocessing")
            return None, None
        
        # Vectorize
        vectorizer = st.session_state.results['vectorizers'][model_type]
        text_vector = vectorizer.transform([clean_text])
        
        if show_debug:
            # Debug: Show vector info
            st.write(f"Vector shape: {text_vector.shape}")
            st.write(f"Vector sum: {text_vector.sum()}")
            st.write(f"Non-zero features: {text_vector.nnz}")
        
        # Predict
        model = st.session_state.results['models'][model_type]
        prediction = model.predict(text_vector)[0]
        probability = model.predict_proba(text_vector)[0]
        
        if show_debug:
            # Debug: Show raw prediction
            st.write(f"Model classes: {model.classes_}")
            st.write(f"Raw probabilities: {probability}")
            st.write(f"Prediction: {prediction}")
            
            # Show which features were found
            feature_names = vectorizer.get_feature_names_out()
            if hasattr(text_vector, 'toarray'):
                vector_array = text_vector.toarray()[0]
                found_features = [(feature_names[i], vector_array[i]) for i in range(len(vector_array)) if vector_array[i] > 0]
                st.write(f"Found features: {found_features[:10]}")  # Show first 10
        
         # ✅ ADDED: Enhanced debugging
        if show_debug:
            # Show which features were found (enhanced)
            feature_names = vectorizer.get_feature_names_out()
            if hasattr(text_vector, 'toarray'):
                vector_array = text_vector.toarray()[0]
                found_features = [(feature_names[i], vector_array[i]) for i in range(len(vector_array)) if vector_array[i] > 0]
                
                # ✅ ADDED: Separate unigrams and bigrams for better analysis
                unigrams = [(f, w) for f, w in found_features if ' ' not in f]
                bigrams = [(f, w) for f, w in found_features if ' ' in f]
                
                st.write(f"Unigrams found: {unigrams}")
                st.write(f"Bigrams found: {bigrams}")  # ✅ Key for "tidak suka"
            
        # Get probability for each class
        classes = model.classes_
        prob_dict = {classes[i]: ensure_numeric(probability[i], 0.0) for i in range(len(classes))}
        
        if show_debug:
            # Debug: Show final probabilities
            st.write(f"Final prob_dict: {prob_dict}")
        
        return prediction, prob_dict
    
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None, None

def safe_text_analysis(df_clean, sentiment_type):
    """Safely extract and prepare text data for analysis"""
    try:
        # Filter by sentiment
        filtered_data = df_clean[df_clean['sentiment'] == sentiment_type]
        
        if len(filtered_data) == 0:
            return None
        
        # Get text column
        if 'review_text_clean' in filtered_data.columns:
            texts = filtered_data['review_text_clean']
        else:
            return None
        
        # Clean and validate texts
        valid_texts = []
        for text in texts:
            if text is not None and not pd.isna(text) and str(text).strip() != '' and str(text) != 'nan':
                clean_text = str(text).strip()
                if len(clean_text) > 2:  # Must have at least some content
                    valid_texts.append(clean_text)
        
        return valid_texts if valid_texts else None
    
    except Exception as e:
        st.warning(f"Error in text analysis preparation: {str(e)}")
        return None

def create_word_frequency_chart(text_data, sentiment, max_words=20):
    """Create word frequency chart sebagai alternatif word cloud"""
    try:
        # Comprehensive input validation
        if text_data is None:
            return None
        
        # Handle different input types (list, pandas Series, numpy array)
        if hasattr(text_data, 'tolist'):
            text_list = text_data.tolist()
        elif isinstance(text_data, (list, tuple)):
            text_list = list(text_data)
        else:
            return None
        
        # Check if empty
        if len(text_list) == 0:
            return None
        
        # Filter out empty/null values more comprehensively
        valid_texts = []
        for text in text_list:
            if text is not None:
                text_str = str(text).strip()
                if text_str and text_str != 'nan' and text_str != 'None' and len(text_str) > 2:
                    valid_texts.append(text_str)
        
        if len(valid_texts) == 0:
            return None
        
        # Gabungkan semua text
        all_text = ' '.join(valid_texts)
        
        if not all_text.strip():
            return None
        
        # Split into words dan hitung frekuensi
        words = all_text.split()
        word_freq = {}
        
        for word in words:
            word = word.strip().lower()  # Normalize to lowercase
            # Filter kata yang terlalu pendek atau mengandung karakter aneh
            if len(word) > 2 and word.isalpha():  # Only alphabetic characters
                word_freq[word] = word_freq.get(word, 0) + 1
        
        if not word_freq:
            return None
        
        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:max_words]
        
        if not sorted_words:
            return None
        
        # Create bar chart
        words_list = [item[0] for item in sorted_words]
        freq_list = [item[1] for item in sorted_words]
        
        color = '#2E8B57' if sentiment == 'Positif' else '#DC143C'
        
        fig = go.Figure(data=[
            go.Bar(
                x=freq_list,
                y=words_list,
                orientation='h',
                marker_color=color,
                text=freq_list,
                textposition='auto',
                hovertemplate='<b>%{y}</b><br>Frekuensi: %{x}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title=f'Top {len(sorted_words)} Kata - Sentiment {sentiment}',
            xaxis_title='Frekuensi',
            yaxis_title='Kata',
            height=max(400, len(sorted_words) * 25),  # Dynamic height
            yaxis={'categoryorder': 'total ascending'},
            font=dict(size=12),
            showlegend=False,
            margin=dict(l=100, r=50, t=50, b=50)  # Better margins
        )
        
        return fig
    
    except Exception as e:
        st.warning(f"Error creating word frequency chart: {str(e)}")
        return None

def create_simple_wordcloud_chart(text_data, sentiment):
    """Create traditional-style word cloud using matplotlib"""
    try:
        # Comprehensive input validation
        if text_data is None:
            return None
        
        # Handle different input types
        if hasattr(text_data, 'tolist'):
            text_list = text_data.tolist()
        elif isinstance(text_data, (list, tuple)):
            text_list = list(text_data)
        else:
            return None
        
        # Check if empty
        if len(text_list) == 0:
            return None
        
        # Filter out empty/null values
        valid_texts = []
        for text in text_list:
            if text is not None:
                text_str = str(text).strip()
                if text_str and text_str != 'nan' and text_str != 'None' and len(text_str) > 2:
                    valid_texts.append(text_str)
        
        if len(valid_texts) == 0:
            return None
        
        # Gabungkan semua text
        all_text = ' '.join(valid_texts)
        
        if not all_text.strip():
            return None
        
        # Hitung frekuensi kata
        words = all_text.split()
        word_freq = {}
        
        for word in words:
            word = word.strip().lower()  # Normalize to lowercase
            if len(word) > 2 and word.isalpha():  # Only alphabetic characters
                word_freq[word] = word_freq.get(word, 0) + 1
        
        if not word_freq:
            return None
        
        # Clear any existing plots
        clear_matplotlib()
        
        # Create figure with white background
        fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
        ax.set_facecolor('white')
        
        # Get top words (limit to prevent overcrowding)
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:40]
        
        if not sorted_words:
            clear_matplotlib()
            return None
        
        # Calculate font sizes based on frequency
        max_freq = sorted_words[0][1]
        min_freq = sorted_words[-1][1]
        
        # Set color scheme based on sentiment
        if sentiment == 'Positif':
            colors = ['#2d5a27', '#3d7a37', '#4d9a47', '#5dba57', '#6dda67']  # Various greens
        else:
            colors = ['#8b0000', '#b22222', '#dc143c', '#ff6347', '#ff7f7f']  # Various reds
        
        # Create word cloud layout using a more sophisticated placement
        placed_words = []
        
        for i, (word, freq) in enumerate(sorted_words):
            # Calculate font size (larger for more frequent words)
            if max_freq == min_freq:
                font_size = 20
            else:
                # Font size range from 12 to 60
                font_size = 12 + (freq - min_freq) / (max_freq - min_freq) * 48
            
            # Choose color based on frequency (darker for more frequent)
            color_idx = min(int((freq - min_freq) / (max_freq - min_freq) * len(colors)), len(colors) - 1)
            color = colors[color_idx]
            
            # Try to place word without overlap
            placed = False
            attempts = 0
            max_attempts = 50
            
            while not placed and attempts < max_attempts:
                # Random position
                if i == 0:  # Place first (most frequent) word in center
                    x = 0.5
                    y = 0.5
                else:
                    # Place other words around center with some randomness
                    angle = np.random.uniform(0, 2 * np.pi)
                    radius = np.random.uniform(0.1, 0.4)
                    x = 0.5 + radius * np.cos(angle)
                    y = 0.5 + radius * np.sin(angle)
                
                # Check if position is within bounds
                if 0.1 <= x <= 0.9 and 0.15 <= y <= 0.85:
                    # Simple overlap check (basic)
                    overlap = False
                    for prev_x, prev_y, prev_size in placed_words:
                        distance = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
                        min_distance = (font_size + prev_size) / 2000  # Rough estimate
                        if distance < min_distance:
                            overlap = True
                            break
                    
                    if not overlap or attempts > 30:  # Accept position if too many attempts
                        placed = True
                        placed_words.append((x, y, font_size))
                
                attempts += 1
            
            # Place the word
            ax.text(x, y, word, fontsize=font_size, color=color, 
                   ha='center', va='center', weight='bold',
                   transform=ax.transAxes,  # Use axes coordinates (0-1)
                   alpha=0.8 + 0.2 * (freq / max_freq))  # More frequent = more opaque
        
        # Set layout
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f'Word Cloud - Sentiment {sentiment}', 
                    fontsize=18, fontweight='bold', pad=20, color='#333333')
        ax.axis('off')
        
        # Remove margins
        plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0.05)
        
        return fig
    
    except Exception as e:
        st.warning(f"Error creating word cloud: {str(e)}")
        clear_matplotlib()
        return None

# Alternative: Try to create real wordcloud if library is available
def create_advanced_wordcloud(text_data, sentiment):
    """Create advanced word cloud using wordcloud library if available"""
    try:
        # Try to import wordcloud library
        from wordcloud import WordCloud as WC
        
        # Process text data
        if hasattr(text_data, 'tolist'):
            text_list = text_data.tolist()
        else:
            text_list = text_data
        
        valid_texts = []
        for text in text_list:
            if text is not None:
                text_str = str(text).strip()
                if text_str and text_str != 'nan' and len(text_str) > 2:
                    valid_texts.append(text_str)
        
        if not valid_texts:
            return None
        
        # Combine all text
        all_text = ' '.join(valid_texts)
        
        # Set colors based on sentiment
        if sentiment == 'Positif':
            colormap = 'Greens'
        else:
            colormap = 'Reds'
        
        # Create WordCloud
        wordcloud = WC(
            width=800, 
            height=400, 
            background_color='white',
            colormap=colormap,
            max_words=50,
            relative_scaling=0.5,
            min_font_size=10,
            max_font_size=80,
            prefer_horizontal=0.7
        ).generate(all_text)
        
        # Clear matplotlib
        clear_matplotlib()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title(f'Word Cloud - Sentiment {sentiment}', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        return fig
        
    except ImportError:
        # Fallback to simple wordcloud if library not available
        return create_simple_wordcloud_chart(text_data, sentiment)
    except Exception as e:
        clear_matplotlib()
        return create_simple_wordcloud_chart(text_data, sentiment)

def create_safe_pie_chart(data, title="Chart"):
    """Create safe pie chart dengan error handling"""
    try:
        if data is None or len(data) == 0:
            st.warning("No data available for chart")
            return None
        
        fig = px.pie(
            values=list(data.values),
            names=list(data.index),
            title=title,
            color_discrete_map={'Positif': '#2E8B57', 'Negatif': '#DC143C'}
        )
        
        fig.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )
        
        fig.update_layout(
            showlegend=True,
            height=400,
            font=dict(size=12)
        )
        
        return fig
    
    except Exception as e:
        st.warning(f"Error creating pie chart: {str(e)}")
        return None

def create_safe_bar_chart(data, title="Chart"):
    """Create safe bar chart dengan error handling"""
    try:
        if data is None or len(data) == 0:
            st.warning("No data available for chart")
            return None
        
        fig = go.Figure()
        
        # Add traces safely
        for i, (model_name, metrics) in enumerate(data.items()):
            color = '#667eea' if i == 0 else '#764ba2'
            
            fig.add_trace(go.Bar(
                name=model_name,
                x=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                y=[
                    ensure_numeric(metrics.get('accuracy', 0), 0),
                    ensure_numeric(metrics.get('precision', 0), 0),
                    ensure_numeric(metrics.get('recall', 0), 0),
                    ensure_numeric(metrics.get('f1', 0), 0)
                ],
                marker_color=color,
                text=[f"{ensure_numeric(metrics.get(m, 0), 0):.3f}" for m in ['accuracy', 'precision', 'recall', 'f1']],
                textposition='auto'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Metrics",
            yaxis_title="Score",
            barmode='group',
            height=400,
            yaxis=dict(range=[0, 1]),
            font=dict(size=12)
        )
        
        return fig
    
    except Exception as e:
        st.warning(f"Error creating bar chart: {str(e)}")
        return None

def create_safe_confusion_matrix(y_true, y_pred, title="Confusion Matrix", cmap='Blues'):
    """Create safe confusion matrix dengan matplotlib"""
    try:
        if len(y_true) == 0 or len(y_pred) == 0:
            st.warning("No data available for confusion matrix")
            return None
        
        # Clear any existing plots
        clear_matplotlib()
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Create heatmap
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap=cmap,
            xticklabels=['Negatif', 'Positif'],
            yticklabels=['Negatif', 'Positif'],
            ax=ax,
            square=True,
            cbar_kws={"shrink": .8}
        )
        
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    except Exception as e:
        st.warning(f"Error creating confusion matrix: {str(e)}")
        clear_matplotlib()
        return None

# ============================================================================
# STREAMLIT APP PAGES
# ============================================================================

def main():
    # Custom CSS yang aman
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .metric-box {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #667eea;
            margin: 0.5rem 0;
        }
        
        .success-box {
            background: linear-gradient(90deg, #2E8B57, #228B22);
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
            text-align: center;
            margin: 1rem 0;
        }
        
        .error-box {
            background: linear-gradient(90deg, #DC143C, #B22222);
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
            text-align: center;
            margin: 1rem 0;
        }
        
        .info-box {
            background: #e3f2fd;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #2196f3;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Analisis Sentiment dengan Naive Bayes</h1>
        <p>Klasifikasi Sentiment Review Pada Simetri Home Coffee & Roastery- Data Mining Implementation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Library status
    with st.expander("📚 Library Status", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if NLTK_AVAILABLE:
                st.success("✅ NLTK: Available")
            else:
                st.warning("⚠️ NLTK: Using fallback")
        
        with col2:
            if SASTRAWI_AVAILABLE:
                st.success("✅ Sastrawi: Available")
            else:
                st.warning("⚠️ Sastrawi: Using fallback")
        
        with col3:
            st.info("📊 ML: Scikit-learn Ready")
    
    # Sidebar navigation
    st.sidebar.markdown("## 📊 Navigation")
    page = st.sidebar.selectbox(
        "Pilih Halaman:",
        ["🏠 Dashboard", "📤 Upload Dataset", "🔬 Metodologi", "📈 Model Performance", "🔮 Prediksi Sentiment Baru"]
    )
    
    # Initialize session state
    if 'df_clean' not in st.session_state:
        st.session_state.df_clean = None
    if 'df_raw' not in st.session_state:
        st.session_state.df_raw = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    
    # Page routing
    if page == "📤 Upload Dataset":
        show_upload_page()
    elif page == "🏠 Dashboard":
        show_dashboard()
    elif page == "🔬 Metodologi":
        show_methodology()
    elif page == "📈 Model Performance":
        show_model_performance()
    elif page == "🔮 Prediksi Sentiment Baru":
        show_prediction_page()
    

def show_upload_page():
    """Halaman upload dataset dengan detail preprocessing dan testing"""
    st.header("📤 Upload Dataset")
    
    st.markdown("""
    Upload dataset Anda atau gunakan sample data untuk analisis sentiment.
    
    **Format yang didukung:**
    - CSV (delimiter: `,` atau `;`)
    - Excel (`.xlsx`, `.xls`)
    
    **Kolom yang dibutuhkan:**
    - `review_text`: Teks review untuk analisis
    - `rating`: Rating numerik (1-5 atau 10-50)
    """)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Pilih file dataset:",
        type=['csv', 'xlsx', 'xls'],
        help="Upload file CSV atau Excel dengan kolom review_text dan rating"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Gunakan Sample Data", type="secondary"):
            with st.spinner("Generating sample data..."):
                try:
                    show_complete_processing_pipeline(generate_sample_data())
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with col2:
        if uploaded_file is not None:
            if st.button("🚀 Proses Dataset", type="primary"):
                try:
                    with st.spinner("Loading dataset..."):
                        df_raw = load_uploaded_data(uploaded_file)
                    
                    if df_raw is not None:
                        show_complete_processing_pipeline(df_raw)
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # Current dataset info
    if st.session_state.df_clean is not None:
        st.markdown("---")
        st.subheader("📊 Dataset Saat Ini")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Samples", len(st.session_state.df_clean))
        
        with col2:
            pos_count = (st.session_state.df_clean['sentiment'] == 'Positif').sum()
            st.metric("Positif", pos_count)
        
        with col3:
            neg_count = (st.session_state.df_clean['sentiment'] == 'Negatif').sum()
            st.metric("Negatif", neg_count)
        
        with col4:
            if st.session_state.results:
                best_acc = max(
                    st.session_state.results['metrics']['bow']['accuracy'],
                    st.session_state.results['metrics']['tfidf']['accuracy']
                )
                st.metric("Best Accuracy", f"{best_acc:.1%}")

def show_complete_processing_pipeline(df_raw):
    """Tampilkan semua tahapan preprocessing sampai testing"""
    
    # ========================================================================
    # STEP 1: DATA LOADING & EXPLORATION
    # ========================================================================
    st.markdown("---")
    st.header("🔍 STEP 1: Data Loading & Exploration")
    
    if df_raw is not None:
        st.success(f"✅ Dataset berhasil dimuat! Shape: {df_raw.shape}")
        
        # Basic info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", len(df_raw))
        with col2:
            st.metric("Total Columns", len(df_raw.columns))
        with col3:
            missing_count = df_raw.isnull().sum().sum()
            st.metric("Missing Values", missing_count)
        
        # Data preview
        with st.expander("👁️ Preview Raw Data", expanded=True):
            st.dataframe(df_raw.head(10), use_container_width=True)
        
        # Column info
        with st.expander("📋 Column Information"):
            st.write("**Columns:**", list(df_raw.columns))
            st.write("**Data Types:**")
            st.dataframe(df_raw.dtypes, use_container_width=True)
        
        # Rating distribution
        if 'rating' in df_raw.columns:
            with st.expander("⭐ Rating Distribution"):
                rating_counts = df_raw['rating'].value_counts().sort_index()
                fig = px.bar(x=rating_counts.index, y=rating_counts.values,
                           title="Distribusi Rating", 
                           labels={'x': 'Rating', 'y': 'Count'})
                st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # STEP 2: DATA PREPROCESSING
    # ========================================================================
    st.markdown("---")
    st.header("🧹 STEP 2: Data Preprocessing")
    
    with st.spinner("Processing data..."):
        df_clean = process_data_with_details(df_raw)
    
    if df_clean is not None:
        st.session_state.df_raw = df_raw
        st.session_state.df_clean = df_clean
        
        st.success(f"✅ Data preprocessing selesai! Valid samples: {len(df_clean)}")
        
        # Show preprocessing results
        show_preprocessing_details(df_raw, df_clean)
        
        # ====================================================================
        # STEP 3: MODEL TRAINING
        # ====================================================================
        st.markdown("---")
        st.header("🤖 STEP 3: Model Training")
        
        with st.spinner("Training models..."):
            results = train_models_with_details(df_clean)
            st.session_state.results = results
        
        if results is not None:
            st.success("✅ Model berhasil dilatih!")
            show_training_details(results)
            
            # ================================================================
            # STEP 4: MODEL EVALUATION  
            # ================================================================
            st.markdown("---")
            st.header("📊 STEP 4: Model Evaluation")
            show_evaluation_details(results)
            
            st.balloons()
            st.success("🎉 Proses data mining selesai! Silakan check Dashboard dan Model Performance.")
        else:
            st.error("❌ Gagal melatih model")

def process_data_with_details(df):
    """Process data dengan detail progress"""
    try:
        st.write("**📋 Checking required columns...**")
        
        # Check required columns
        required_cols = ['review_text', 'rating']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"❌ Kolom yang dibutuhkan tidak ditemukan: {missing_cols}")
            return None
        else:
            st.success("✅ Kolom yang dibutuhkan tersedia")
        
        st.write("**🧹 Data cleaning...**")
        
        # Data cleaning dengan progress
        original_count = len(df)
        df_clean = df.dropna(subset=['review_text']).copy()
        after_dropna = len(df_clean)
        
        df_clean = df_clean[df_clean['review_text'].astype(str).str.strip() != '']
        after_empty_removal = len(df_clean)
        
        st.write(f"- Original data: {original_count} rows")
        st.write(f"- After dropping NaN: {after_dropna} rows (-{original_count - after_dropna})")
        st.write(f"- After removing empty text: {after_empty_removal} rows (-{after_dropna - after_empty_removal})")
        
        if len(df_clean) == 0:
            st.error("❌ Tidak ada data valid setelah cleaning")
            return None
        
        st.write("**🏷️ Creating sentiment labels...**")
        
        # Create sentiment labels
        df_clean['sentiment'] = df_clean['rating'].apply(create_sentiment_label)
        
        # Show sentiment distribution
        sentiment_counts = df_clean['sentiment'].value_counts()
        st.write("Distribusi sentiment:")
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(df_clean)) * 100
            st.write(f"- {sentiment}: {count} ({percentage:.1f}%)")
        
        st.write("**📝 Text preprocessing...**")
        
        # Text preprocessing dengan progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        processed_texts = []
        for i, text in enumerate(df_clean['review_text']):
            processed_text = preprocess_text(text)
            processed_texts.append(processed_text)
            
            # Update progress every 10%
            if i % max(1, len(df_clean) // 10) == 0:
                progress = (i + 1) / len(df_clean)
                progress_bar.progress(progress)
                status_text.text(f'Processing text {i+1}/{len(df_clean)}')
        
        df_clean['review_text_clean'] = processed_texts
        
        # Remove empty texts after preprocessing
        before_clean = len(df_clean)
        df_clean = df_clean[df_clean['review_text_clean'].astype(str).str.strip() != '']
        after_clean = len(df_clean)
        
        progress_bar.empty()
        status_text.empty()
        
        st.write(f"- Text setelah preprocessing: {after_clean} valid texts (-{before_clean - after_clean} empty)")
        
        if len(df_clean) == 0:
            st.error("❌ Tidak ada data valid setelah preprocessing")
            return None
        
        return df_clean
    
    except Exception as e:
        st.error(f"❌ Error processing data: {str(e)}")
        return None

def show_preprocessing_details(df_raw, df_clean):
    """Show detailed preprocessing results"""
    
    # Preprocessing summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Original Samples", len(df_raw))
    with col2:
        st.metric("After Cleaning", len(df_clean))
    with col3:
        data_loss = ((len(df_raw) - len(df_clean)) / len(df_raw)) * 100
        st.metric("Data Loss", f"{data_loss:.1f}%")
    
    # Sample preprocessing examples
    with st.expander("📝 Sample Preprocessing Results", expanded=True):
        st.write("**Before vs After Preprocessing:**")
        
        sample_size = min(5, len(df_clean))
        for i in range(sample_size):
            original = df_clean.iloc[i]['review_text']
            processed = df_clean.iloc[i]['review_text_clean']
            sentiment = df_clean.iloc[i]['sentiment']
            
            st.write(f"**Sample {i+1} - {sentiment}:**")
            st.write(f"Original: {original[:100]}...")
            st.write(f"Processed: {processed[:100]}...")
            st.write("---")
    
    # Text statistics
    with st.expander("📊 Text Statistics"):
        original_lengths = df_clean['review_text'].str.len()
        processed_lengths = df_clean['review_text_clean'].str.len()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Original Text Lengths:**")
            st.write(f"- Mean: {original_lengths.mean():.1f}")
            st.write(f"- Min: {original_lengths.min()}")
            st.write(f"- Max: {original_lengths.max()}")
        
        with col2:
            st.write("**Processed Text Lengths:**")
            st.write(f"- Mean: {processed_lengths.mean():.1f}")
            st.write(f"- Min: {processed_lengths.min()}")
            st.write(f"- Max: {processed_lengths.max()}")

def train_models_with_details(df_clean):
    """Train models dengan detailed output"""
    try:
        st.write("**📊 Preparing data for training...**")
        
        # Prepare data
        X_text = df_clean['review_text_clean']
        y = df_clean['sentiment']
        
        # Show class distribution
        class_counts = y.value_counts()
        st.write("Class distribution:")
        for class_name, count in class_counts.items():
            percentage = (count / len(y)) * 100
            st.write(f"- {class_name}: {count} ({percentage:.1f}%)")
        
        if len(class_counts) < 2:
            st.error("❌ Dataset harus memiliki minimal 2 kelas sentiment")
            return None
        
        st.write("**🔧 Vectorization...**")
        
        # Vectorization
        bow_vectorizer = CountVectorizer(
            max_features=min(200, len(X_text)),
            min_df=1,
            max_df=0.9,
            ngram_range=(1, 1),
            lowercase=True,
            strip_accents='unicode',
            token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b'
        )
        X_bow = bow_vectorizer.fit_transform(X_text)
        
        tfidf_vectorizer = TfidfVectorizer(
            max_features=min(200, len(X_text)),
            min_df=1,
            max_df=0.9,
            ngram_range=(1, 1),
            lowercase=True,
            strip_accents='unicode',
            token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b'
        )
        X_tfidf = tfidf_vectorizer.fit_transform(X_text)
        
        st.write(f"- Bag of Words matrix: {X_bow.shape}")
        st.write(f"- TF-IDF matrix: {X_tfidf.shape}")
        st.write(f"- Vocabulary size: {len(bow_vectorizer.vocabulary_)}")
        
        st.write("**🔄 Train-test split...**")
        
        # Split data
        test_size = 0.2
        try:
            X_bow_train, X_bow_test, y_train, y_test = train_test_split(
                X_bow, y, test_size=test_size, random_state=42, stratify=y
            )
            X_tfidf_train, X_tfidf_test, _, _ = train_test_split(
                X_tfidf, y, test_size=test_size, random_state=42, stratify=y
            )
            st.write("✅ Stratified split successful")
        except ValueError:
            X_bow_train, X_bow_test, y_train, y_test = train_test_split(
                X_bow, y, test_size=test_size, random_state=42
            )
            X_tfidf_train, X_tfidf_test, _, _ = train_test_split(
                X_tfidf, y, test_size=test_size, random_state=42
            )
            st.write("⚠️ Using random split (stratification failed)")
        
        st.write(f"- Training samples: {len(y_train)}")
        st.write(f"- Test samples: {len(y_test)}")
        
        # Show split distribution
        train_dist = y_train.value_counts()
        test_dist = y_test.value_counts()
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Training set:**")
            for class_name, count in train_dist.items():
                percentage = (count / len(y_train)) * 100
                st.write(f"- {class_name}: {count} ({percentage:.1f}%)")
        
        with col2:
            st.write("**Test set:**")
            for class_name, count in test_dist.items():
                percentage = (count / len(y_test)) * 100
                st.write(f"- {class_name}: {count} ({percentage:.1f}%)")
        
        st.write("**🧠 Training Naive Bayes models...**")
        
        # Train models
        nb_bow = MultinomialNB(alpha=1.0)
        nb_bow.fit(X_bow_train, y_train)
        y_pred_bow = nb_bow.predict(X_bow_test)
        
        nb_tfidf = MultinomialNB(alpha=1.0)
        nb_tfidf.fit(X_tfidf_train, y_train)
        y_pred_tfidf = nb_tfidf.predict(X_tfidf_test)
        
        st.write("✅ Model training completed")
        
        # Calculate metrics
        metrics_bow = safe_metrics_calculation(y_test, y_pred_bow)
        metrics_tfidf = safe_metrics_calculation(y_test, y_pred_tfidf)
        
        return {
            'models': {'bow': nb_bow, 'tfidf': nb_tfidf},
            'vectorizers': {'bow': bow_vectorizer, 'tfidf': tfidf_vectorizer},
            'metrics': {'bow': metrics_bow, 'tfidf': metrics_tfidf},
            'predictions': {'bow': y_pred_bow, 'tfidf': y_pred_tfidf},
            'test_data': {'y_test': y_test},
            'train_info': {
                'total_samples': len(df_clean),
                'train_samples': len(y_train),
                'test_samples': len(y_test),
                'classes': list(y.unique()),
                'class_distribution': dict(y.value_counts()),
                'feature_names': {
                    'bow': bow_vectorizer.get_feature_names_out()[:50].tolist(),
                    'tfidf': tfidf_vectorizer.get_feature_names_out()[:50].tolist()
                }
            }
        }
    
    except Exception as e:
        st.error(f"❌ Error training models: {str(e)}")
        return None

def show_training_details(results):
    """Show detailed training results"""
    
    # Model performance overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**🔵 Bag of Words Model:**")
        metrics = results['metrics']['bow']
        st.write(f"- Accuracy: {metrics['accuracy']:.4f}")
        st.write(f"- Precision: {metrics['precision']:.4f}")
        st.write(f"- Recall: {metrics['recall']:.4f}")
        st.write(f"- F1-Score: {metrics['f1']:.4f}")
    
    with col2:
        st.write("**🔴 TF-IDF Model:**")
        metrics = results['metrics']['tfidf']
        st.write(f"- Accuracy: {metrics['accuracy']:.4f}")
        st.write(f"- Precision: {metrics['precision']:.4f}")
        st.write(f"- Recall: {metrics['recall']:.4f}")
        st.write(f"- F1-Score: {metrics['f1']:.4f}")
    
    # Best model
    best_model = 'TF-IDF' if results['metrics']['tfidf']['accuracy'] > results['metrics']['bow']['accuracy'] else 'Bag of Words'
    best_accuracy = max(results['metrics']['tfidf']['accuracy'], results['metrics']['bow']['accuracy'])
    
    st.success(f"🏆 **Best Model**: {best_model} dengan akurasi **{best_accuracy:.1%}**")
    
    # Sample vocabulary
    with st.expander("📚 Sample Vocabulary Learned"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Bag of Words Features:**")
            bow_features = results['train_info']['feature_names']['bow']
            st.write(", ".join(bow_features[:30]))
        
        with col2:
            st.write("**TF-IDF Features:**")
            tfidf_features = results['train_info']['feature_names']['tfidf']
            st.write(", ".join(tfidf_features[:30]))

def show_evaluation_details(results):
    """Show detailed evaluation results"""
    
    # Confusion matrices
    st.write("**📊 Confusion Matrices:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Bag of Words Model**")
        try:
            y_test = results['test_data']['y_test']
            y_pred_bow = results['predictions']['bow']
            
            fig = create_safe_confusion_matrix(y_test, y_pred_bow, "Confusion Matrix - BoW", 'Blues')
            if fig:
                st.pyplot(fig)
                clear_matplotlib()
        except Exception as e:
            st.error(f"Error creating BoW confusion matrix: {str(e)}")
    
    with col2:
        st.write("**TF-IDF Model**")
        try:
            y_pred_tfidf = results['predictions']['tfidf']
            
            fig = create_safe_confusion_matrix(y_test, y_pred_tfidf, "Confusion Matrix - TF-IDF", 'Greens')
            if fig:
                st.pyplot(fig)
                clear_matplotlib()
        except Exception as e:
            st.error(f"Error creating TF-IDF confusion matrix: {str(e)}")
    
    # Classification reports
    with st.expander("📋 Classification Reports"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Bag of Words:**")
            try:
                report_bow = classification_report(y_test, y_pred_bow, output_dict=True)
                report_df_bow = pd.DataFrame(report_bow).transpose()
                st.dataframe(report_df_bow.round(4))
            except Exception as e:
                st.error(f"Error generating BoW report: {str(e)}")
        
        with col2:
            st.write("**TF-IDF:**")
            try:
                report_tfidf = classification_report(y_test, y_pred_tfidf, output_dict=True)
                report_df_tfidf = pd.DataFrame(report_tfidf).transpose()
                st.dataframe(report_df_tfidf.round(4))
            except Exception as e:
                st.error(f"Error generating TF-IDF report: {str(e)}")
    
    # Model comparison chart
    st.write("**📈 Model Comparison Chart:**")
    try:
        model_data = {
            'Bag of Words': results['metrics']['bow'],
            'TF-IDF': results['metrics']['tfidf']
        }
        fig = create_safe_bar_chart(model_data, "Model Performance Comparison")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating comparison chart: {str(e)}")

def show_dashboard():
    """Dashboard utama dengan visualisasi yang aman dan word cloud"""
    if st.session_state.df_clean is None:
        st.warning("⚠️ Belum ada dataset. Silakan upload di halaman Upload Dataset.")
        return
    
    if st.session_state.results is None:
        st.error("❌ Model belum dilatih. Silakan upload ulang dataset.")
        return
    
    st.header("📊 Dashboard Overview")
    
    df_clean = st.session_state.df_clean
    results = st.session_state.results
    
    # Metrics overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Review", len(df_clean))
    
    with col2:
        best_accuracy = max(
            results['metrics']['bow']['accuracy'],
            results['metrics']['tfidf']['accuracy']
        )
        st.metric("Best Accuracy", f"{best_accuracy:.1%}")
    
    with col3:
        pos_count = (df_clean['sentiment'] == 'Positif').sum()
        st.metric("Review Positif", pos_count)
    
    with col4:
        neg_count = (df_clean['sentiment'] == 'Negatif').sum()
        st.metric("Review Negatif", neg_count)
    
    # Charts Row 1: Sentiment Distribution and Model Performance
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Distribusi Sentiment")
        try:
            sentiment_counts = df_clean['sentiment'].value_counts()
            fig = create_safe_pie_chart(sentiment_counts, "Distribusi Sentiment Review")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating sentiment chart: {str(e)}")
    
    with col2:
        st.subheader("📈 Performa Model")
        try:
            model_data = {
                'Bag of Words': results['metrics']['bow'],
                'TF-IDF': results['metrics']['tfidf']
            }
            fig = create_safe_bar_chart(model_data, "Perbandingan Performa Model")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating performance chart: {str(e)}")
    
    # Word Analysis Section
    st.subheader("☁️ Analisis Kata per Sentiment")
    
    # Word frequency charts for each sentiment
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Kata-kata Sentiment Positif**")
        try:
            positive_texts = safe_text_analysis(df_clean, 'Positif')
            if positive_texts and len(positive_texts) > 0:
                fig = create_word_frequency_chart(positive_texts, 'Positif', max_words=15)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Tidak ada kata yang cukup untuk analisis positif")
            else:
                st.info("Tidak ada review positif yang valid")
        except Exception as e:
            st.error(f"Error analyzing positive words: {str(e)}")
    
    with col2:
        st.write("**Kata-kata Sentiment Negatif**")
        try:
            negative_texts = safe_text_analysis(df_clean, 'Negatif')
            if negative_texts and len(negative_texts) > 0:
                fig = create_word_frequency_chart(negative_texts, 'Negatif', max_words=15)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Tidak ada kata yang cukup untuk analisis negatif")
            else:
                st.info("Tidak ada review negatif yang valid")
        except Exception as e:
            st.error(f"Error analyzing negative words: {str(e)}")
    
    # Alternative: Word Clouds using advanced algorithm
    st.subheader("☁️ Word Cloud Visualization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Word Cloud - Sentiment Positif**")
        try:
            positive_texts = safe_text_analysis(df_clean, 'Positif')
            if positive_texts and len(positive_texts) > 0:
                # Try advanced wordcloud first, fallback to simple
                fig = create_advanced_wordcloud(positive_texts, 'Positif')
                if fig:
                    st.pyplot(fig)
                    clear_matplotlib()
                else:
                    st.info("Tidak dapat membuat word cloud positif")
            else:
                st.info("Tidak ada teks positif yang valid untuk word cloud")
        except Exception as e:
            st.error(f"Error creating positive word cloud: {str(e)}")
            clear_matplotlib()
    
    with col2:
        st.write("**Word Cloud - Sentiment Negatif**")
        try:
            negative_texts = safe_text_analysis(df_clean, 'Negatif')
            if negative_texts and len(negative_texts) > 0:
                # Try advanced wordcloud first, fallback to simple
                fig = create_advanced_wordcloud(negative_texts, 'Negatif')
                if fig:
                    st.pyplot(fig)
                    clear_matplotlib()
                else:
                    st.info("Tidak dapat membuat word cloud negatif")
            else:
                st.info("Tidak ada teks negatif yang valid untuk word cloud")
        except Exception as e:
            st.error(f"Error creating negative word cloud: {str(e)}")
            clear_matplotlib()
    
    # Training info
    if 'train_info' in results:
        st.subheader("🔧 Training Information")
        info = results['train_info']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", info['total_samples'])
        with col2:
            st.metric("Training Samples", info['train_samples'])
        with col3:
            st.metric("Test Samples", info['test_samples'])
        
        # Class distribution
        if 'class_distribution' in info:
            st.write("**Distribusi Kelas:**")
            for class_name, count in info['class_distribution'].items():
                percentage = (count / info['total_samples']) * 100
                st.write(f"- {class_name}: {count} samples ({percentage:.1f}%)")
        
        # Sample vocabulary
        if 'feature_names' in info:
            with st.expander("📝 Sample Vocabulary yang Dipelajari Model"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Bag of Words Features:**")
                    bow_features = info['feature_names'].get('bow', [])
                    st.write(", ".join(bow_features[:20]) if bow_features else "No features")
                with col2:
                    st.write("**TF-IDF Features:**")
                    tfidf_features = info['feature_names'].get('tfidf', [])
                    st.write(", ".join(tfidf_features[:20]) if tfidf_features else "No features")

def show_methodology():
    """Halaman metodologi"""
    st.header("🔬 Metodologi Data Mining - 5 Langkah")
    
    st.markdown("""
    ## 🔄 5 Tahapan Data Mining Process
    
    Aplikasi ini mengimplementasikan metodologi **CRISP-DM** (Cross-Industry Standard Process for Data Mining) 
    yang telah diadaptasi untuk analisis sentiment menggunakan algoritma **Naive Bayes**.
    """)
    
    steps = [
        {
            "title": "1️⃣ Selection (Seleksi Data)",
            "description": "Pengumpulan dan eksplorasi dataset review",
            "content": """
            - **Dataset**: Upload file CSV/Excel atau gunakan sample data
            - **Eksplorasi**: Analisis struktur data dan distribusi rating
            - **Validasi**: Cek kolom yang diperlukan (review_text, rating)
            - **Quality Check**: Identifikasi missing values dan outliers
            """
        },
        {
            "title": "2️⃣ Preprocessing (Pra-pemrosesan Data)",
            "description": "Pembersihan dan normalisasi teks bahasa Indonesia",
            "content": """
            - **Text Cleaning**: Hapus URL, emoji, karakter khusus
            - **Tokenisasi**: Pecah teks menjadi kata-kata individual
            - **Stopword Removal**: Hapus kata-kata umum bahasa Indonesia
            - **Stemming**: Ubah kata ke bentuk dasar (jika Sastrawi tersedia)
            - **Label Creation**: Konversi rating ke sentiment (Positif/Negatif)
            """
        },
        {
            "title": "3️⃣ Transformation (Transformasi Data)",
            "description": "Konversi teks ke representasi numerik",
            "content": """
            - **Bag of Words**: Hitung frekuensi kemunculan kata
            - **TF-IDF**: Bobot term frequency-inverse document frequency
            - **Feature Selection**: Pilih fitur terbaik (max 500 features)
            - **Vectorization**: Ubah teks menjadi matriks numerik
            """
        },
        {
            "title": "4️⃣ Data Mining (Naive Bayes Algorithm)",
            "description": "Implementasi algoritma klasifikasi probabilistik",
            "content": """
            - **Prior Probability**: Hitung P(Class) untuk setiap sentiment
            - **Likelihood**: Hitung P(Feature|Class) untuk setiap kata
            - **Posterior**: Hitung P(Class|Feature) menggunakan Bayes theorem
            - **Laplace Smoothing**: Handle zero probability dengan alpha=1.0
            - **Training**: Latih model dengan data training (80%)
            """
        },
        {
            "title": "5️⃣ Evaluation & Interpretation",
            "description": "Evaluasi model dan interpretasi hasil",
            "content": """
            - **Test Set**: Evaluasi dengan data testing (20%)
            - **Metrics**: Hitung Accuracy, Precision, Recall, F1-Score
            - **Confusion Matrix**: Visualisasi prediksi vs aktual
            - **Model Comparison**: Bandingkan BoW vs TF-IDF
            - **Prediction**: Test dengan data baru dan interpretasi
            """
        }
    ]
    
    for step in steps:
        with st.expander(step["title"], expanded=False):
            st.markdown(f"**{step['description']}**")
            st.markdown(step["content"])

def show_model_performance():
    """Halaman performa model dengan visualisasi yang aman"""
    if st.session_state.results is None:
        st.warning("⚠️ Model belum dilatih. Upload dataset terlebih dahulu.")
        return
    
    st.header("📈 Model Performance Analysis")
    
    results = st.session_state.results
    
    # Model comparison table
    st.subheader("⚖️ Perbandingan Model")
    
    try:
        comparison_data = {
            'Model': ['Naive Bayes + Bag of Words', 'Naive Bayes + TF-IDF'],
            'Accuracy': [
                f"{results['metrics']['bow']['accuracy']:.4f}",
                f"{results['metrics']['tfidf']['accuracy']:.4f}"
            ],
            'Precision': [
                f"{results['metrics']['bow']['precision']:.4f}",
                f"{results['metrics']['tfidf']['precision']:.4f}"
            ],
            'Recall': [
                f"{results['metrics']['bow']['recall']:.4f}",
                f"{results['metrics']['tfidf']['recall']:.4f}"
            ],
            'F1-Score': [
                f"{results['metrics']['bow']['f1']:.4f}",
                f"{results['metrics']['tfidf']['f1']:.4f}"
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Best model
        best_model = 'TF-IDF' if results['metrics']['tfidf']['accuracy'] > results['metrics']['bow']['accuracy'] else 'Bag of Words'
        best_accuracy = max(results['metrics']['tfidf']['accuracy'], results['metrics']['bow']['accuracy'])
        
        st.success(f"🏆 **Model Terbaik**: Naive Bayes + {best_model} dengan akurasi **{best_accuracy:.1%}**")
        
    except Exception as e:
        st.error(f"Error displaying comparison: {str(e)}")
    
    # Confusion matrices
    try:
        if len(results['test_data']['y_test']) > 0:
            st.subheader("📊 Confusion Matrix")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Bag of Words Model**")
                y_test = results['test_data']['y_test']
                y_pred_bow = results['predictions']['bow']
                
                fig = create_safe_confusion_matrix(y_test, y_pred_bow, "Confusion Matrix - BoW", 'Blues')
                if fig:
                    st.pyplot(fig)
                    clear_matplotlib()
            
            with col2:
                st.write("**TF-IDF Model**")
                y_pred_tfidf = results['predictions']['tfidf']
                
                fig = create_safe_confusion_matrix(y_test, y_pred_tfidf, "Confusion Matrix - TF-IDF", 'Greens')
                if fig:
                    st.pyplot(fig)
                    clear_matplotlib()
    
    except Exception as e:
        st.error(f"Error creating confusion matrices: {str(e)}")
        clear_matplotlib()

def show_prediction_page():
    """Halaman prediksi sentiment baru dengan debugging"""
    if st.session_state.results is None:
        st.warning("⚠️ Model belum dilatih. Upload dataset terlebih dahulu.")
        return
    
    st.header("🔮 Prediksi Sentiment Baru")
    
    # Model selection
    model_choice = st.radio(
        "Pilih model:",
        ["TF-IDF (Recommended)", "Bag of Words"],
        help="TF-IDF umumnya memberikan hasil yang lebih akurat"
    )
    
    model_type = 'tfidf' if 'TF-IDF' in model_choice else 'bow'
    
    # Show model info
    if st.session_state.results:
        st.subheader("📊 Model Information")
        train_info = st.session_state.results.get('train_info', {})
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Training Samples", train_info.get('train_samples', 'N/A'))
        with col2:
            class_dist = train_info.get('class_distribution', {})
            st.write("**Class Distribution:**")
            for cls, count in class_dist.items():
                st.write(f"- {cls}: {count}")
        with col3:
            metrics = st.session_state.results['metrics'][model_type]
            st.metric("Model Accuracy", f"{metrics['accuracy']:.1%}")
    
    # Text input
    st.subheader("💬 Input Review")
    
    # Test cases yang jelas berbeda
    test_cases = {
        "Sangat Positif": "Enak banget mantap recommended puas suka terbaik amazing excellent nyaman cozy",
        "Sangat Negatif": "Buruk jelek mengecewakan tidak suka zonk pahit mahal lambat kotor berisik",
        "Positif Normal": "Kopi di sini enak banget! Tempat nyaman dan pelayanan ramah sekali.",
        "Negatif Normal": "Kopinya pahit dan tempat terlalu berisik. Pelayanan lambat sekali tidak recommended.",
        "Mixed Review": "Tempat nyaman dan makanan enak, tapi harga agak mahal dan pelayanan lambat.",
        "Neutral": "Tempat kopi biasa, tidak istimewa tapi tidak buruk juga."
    }
    
    st.write("**Test Cases untuk Debugging:**")
    selected_test = st.selectbox("Pilih test case:", [""] + list(test_cases.keys()))
    
    if selected_test:
        text_input = test_cases[selected_test]
    else:
        text_input = ""
    
    text_input = st.text_area(
        "Review text:",
        value=text_input,
        height=120,
        placeholder="Masukkan review untuk dianalisis..."
    )
    
    # Show debugging toggle
    show_debug = st.checkbox("Show Debug Information", value=True)
    
    # Prediction
    if st.button("🔍 Prediksi Sentiment", type="primary"):
        if text_input.strip():
            try:
                with st.spinner("Menganalisis sentiment..."):
                    # Temporary modify predict function to control debug output
                    old_debug = st.session_state.get('show_debug', True)
                    st.session_state.show_debug = show_debug
                    
                    prediction, prob_dict = predict_sentiment(text_input, model_type)
                    
                    st.session_state.show_debug = old_debug
                
                if prediction and prob_dict:
                    st.subheader("📊 Hasil Prediksi Sentiment")
                    
                    # Main result
                    confidence = prob_dict.get(prediction, 0.0)
                    
                    if prediction == 'Positif':
                        st.markdown(f"""
                        <div class="success-box">
                            😊 <strong>Sentiment: {prediction}</strong><br>
                            Confidence: {confidence:.1%}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="error-box">
                            😞 <strong>Sentiment: {prediction}</strong><br>
                            Confidence: {confidence:.1%}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Probability breakdown
                    st.subheader("📈 Distribusi Probabilitas")
                    
                    # Create two columns for probabilities
                    col1, col2 = st.columns(2)
                    
                    prob_items = list(prob_dict.items())
                    for i, (sentiment, prob) in enumerate(prob_items):
                        with col1 if i == 0 else col2:
                            st.metric(f"Probabilitas {sentiment}", f"{prob:.1%}")
                            
                            # Color-coded progress bar
                            if sentiment == 'Positif':
                                st.markdown(f"""
                                <div style="background: #e8f5e8; border-radius: 5px; padding: 5px;">
                                    <div style="background: #2E8B57; height: 20px; width: {prob*100}%; border-radius: 3px; text-align: center; color: white; font-weight: bold;">
                                        {prob:.1%}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div style="background: #ffeaea; border-radius: 5px; padding: 5px;">
                                    <div style="background: #DC143C; height: 20px; width: {prob*100}%; border-radius: 3px; text-align: center; color: white; font-weight: bold;">
                                        {prob:.1%}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Quick test all models
                    st.subheader("🔄 Perbandingan Model")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Bag of Words:**")
                        bow_pred, bow_prob = predict_sentiment_simple(text_input, 'bow')
                        if bow_pred and bow_prob:
                            st.write(f"Prediksi: {bow_pred}")
                            for sent, prob in bow_prob.items():
                                st.write(f"- {sent}: {prob:.1%}")
                    
                    with col2:
                        st.write("**TF-IDF:**")
                        tfidf_pred, tfidf_prob = predict_sentiment_simple(text_input, 'tfidf')
                        if tfidf_pred and tfidf_prob:
                            st.write(f"Prediksi: {tfidf_pred}")
                            for sent, prob in tfidf_prob.items():
                                st.write(f"- {sent}: {prob:.1%}")
                    
                    # Interpretation
                    st.subheader("💡 Interpretasi")
                    
                    if confidence > 0.8:
                        level = "Sangat Tinggi"
                        desc = "Model sangat yakin dengan prediksi ini."
                    elif confidence > 0.6:
                        level = "Tinggi"
                        desc = "Model cukup yakin dengan prediksi ini."
                    elif confidence > 0.5:
                        level = "Sedang"
                        desc = "Prediksi dengan tingkat kepercayaan sedang."
                    else:
                        level = "Rendah"
                        desc = "Prediksi dengan tingkat kepercayaan rendah."
                    
                    st.info(f"""
                    **Model**: {model_choice}  
                    **Sentiment**: {prediction}  
                    **Confidence Level**: {level} ({confidence:.1%})
                    
                    {desc}
                    """)
                
                else:
                    st.error("❌ Gagal melakukan prediksi.")
            
            except Exception as e:
                st.error(f"Error dalam prediksi: {str(e)}")
                import traceback
                st.error(f"Traceback: {traceback.format_exc()}")
        else:
            st.warning("⚠️ Silakan masukkan teks untuk diprediksi!")

def predict_sentiment_simple(text, model_type='tfidf'):
    """Prediksi sentiment tanpa debug output"""
    try:
        if 'results' not in st.session_state or st.session_state.results is None:
            return None, None
        
        # Preprocess text
        clean_text = preprocess_text(text)
        if not clean_text or clean_text.strip() == "":
            return None, None
        
        # Vectorize
        vectorizer = st.session_state.results['vectorizers'][model_type]
        text_vector = vectorizer.transform([clean_text])
        
        # Predict
        model = st.session_state.results['models'][model_type]
        prediction = model.predict(text_vector)[0]
        probability = model.predict_proba(text_vector)[0]
        
        # Get probability for each class
        classes = model.classes_
        prob_dict = {classes[i]: ensure_numeric(probability[i], 0.0) for i in range(len(classes))}
        
        return prediction, prob_dict
    
    except Exception as e:
        return None, None

if __name__ == "__main__":
    main()

