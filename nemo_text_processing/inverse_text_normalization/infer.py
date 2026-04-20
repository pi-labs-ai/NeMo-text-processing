from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer
import time

def run_panjabi_itn():
    # Initialize the InverseNormalizer for Panjabi ('pa')
    try:
        print("Loading Inverse Normalizer for Panjabi (ਪੰਜਾਬੀ)...")
        normalizer = InverseNormalizer(lang='pa')
        
        # Panjabi test cases - Spoken -> Written format
        test_texts = [
            "নয় নয়শো সাতান্ন কোটি আটানব্বই লক্ষ চার হাজার",
            "ਚਾਰਟਾ ਪੈਂਤ੍ਰੀਸੇ ਖੇਤੇ ਗਿਯੇਛਿਲਾਮ। ਪੌਣੇ ਪਾਂਚਟਾਰ ਮੋੱਧੇ ਜੇਤੇ ਹੋਬੇ",
            "ਆਜਕੇਰ ਤਾਰਿਖ ਆਠਾਰੋ ਮਾਰਚ ਦੁਈ ਹਾਜਾਰ ਛਾਬ্বੀਸ਼। ਮਾਨੇ ਆਗਾਮੀਕਾਲ ਉਨੀਸ਼ ਮਾਰਚ ਹੋਬੇ। ਅਥਬਾ ਕੇਉ ਏਕੇ ਮਾਰਚ ਆਠਾਰੋ ਦੁਈ ਹਾਜਾਰ ਛਾਬ্বੀਸ਼ਓ ਬੋਲਤੇ ਪਾਰੇ",
            "ਆਮਾਰ ਆਧਾਰ ਕਾਰਡ ਨੰਬਰ ਏਕਸ਼ੋੱਟੀ ਚੁਆੱਲੀਸ਼ ਚੁਆੱਤਰ ਸਾਤਾਸ਼ ਦੁਈ ਹਾਜਾਰ ਏਕਸ਼ੋ ਸਾਤੱਤਰ ਏਬੋਂ ਮੋਬਾਈਲ ਨੰਬਰ ਏਕ ਦੁਈ ਤੀਨ ਚਾਰ ਪਾਂਚ ਛੋਏ ਸਾਤ ਆਠ ਨੌ ਸ਼ੂਨ੍ਯ",
            "ਆਮਾਰ ਆਧਾਰ ਕਾਰਡ ਨੰਬਰ ਸਿਕਸਟੀ ਫੋਰਟੀ ਫੋਰ ਸੇਵੈਂਟੀ ਟਵੈਂਟੀ ਸੇਵਨ ਟੂ ਥਾਉਜ਼ੈਂਡ ਵਨ ਹੰਡਰਡ ਐਂਡ ਸੇਵੈਂਟੀ ਏਟ ਏਬੋਂ ਮੋਬਾਈਲ ਨੰਬਰ ਵਨ ਟੂ ਥ੍ਰੀ ਫੋਰ ਫਾਈਵ ਸਿਕਸ ਸੇਵਨ ਐਟ ਨਾਈਨ ਜ਼ੀਰੋ",
            "ਆਮਾਰ ਆਧਾਰ ਕਾਰਡ ਨੰਬਰ ਸ਼ਾਠ ਪੈਂਤਾਲੀਸ਼ ਸੱਤਰ ਸਾਤਾਸ਼ ਦੁਈ ਹਾਜਾਰ ਏਕਸ਼ੋ ਆਟਾਤ্তਰ ਏਬੋਂ ਮੋਬਾਈਲ ਨੰਬਰ ਏਕ ਦੁਈ ਤੀਨ ਚਾਰ ਪਾਂਚ ਛੋਏ ਸਾਤ ਆਠ ਨੌ ਸ਼ੂਨ੍ਯ"

            # "अंमली पदार्थांचे सेवन आरोग्यासाठी हानिकारक ठरते.",
            # "अंमली पदार्थांचे सेवन (7) आरोग्यासाठी हानिकारक ठरते.",
            # "तो पावणे बाराला पोहोचला, आणि बारा पाचला जेवायला गेला.",
            # "नवव्या मजल्यावरून व्हिव चांगला दिसतो. पाचवा पन चालेल.",
            # "BMW इस अ गूड कार.",
            # "श्रीमान अक्षय गोडबोले.",
            # "श्री. अक्षय गोडबोले यांनी एक पाव केक आणला.",
            # "श्री. अक्षय गोडबोले यांनी 1/4 एक पाव केक आणला.",
            # "नाईंटी थ्री फोर हंड्रेड अँड सिक्सटी नाईन.",


            # Cardinal numbers
            # "ਇੱਕ ਸੌ ਤੇਈ",  # 123
            # "ਦੋ ਹਜ਼ਾਰ ਚੌਵੀ",  # 2024
            # "ਪੰਜ ਲੱਖ",  # 500000
            
            # # Time expressions  
            # "ਚਾਰ ਵਜੇ ਪੈਂਤੀ ਮਿੰਟ",  # 4:35
            # "ਸਾਢੇ ਪੰਜ ਵਜੇ",  # 5:30
            # "ਪੌਣੇ ਚਾਰ ਵਜੇ",  # 3:45
            
            # # Date expressions
            # "ਅਠਾਰਾਂ ਮਾਰਚ ਦੋ ਹਜ਼ਾਰ ਛੱਬੀ",  # 18 March 2026
            # "ਪੰਜ ਜਨਵਰੀ ਦੋ ਹਜ਼ਾਰ ਬਾਰਾਂ",  # 5 January 2012
            
            # # Telephone numbers
            # "ਮੋਬਾਈਲ ਨੰਬਰ ਇੱਕ ਦੋ ਤਿੰਨ ਚਾਰ ਪੰਜ ਛੇ ਸੱਤ ਅੱਠ ਨੌ ਸਿਫਰ",
            
            # # Decimal
            # "ਇੱਕ ਦਸ਼ਮਲਵ ਪੰਜ",  # 1.5
            
            # # User-requested test cases (note: some have Bengali mixed in)
            # # "নয় নয়শো সাতান্ন কোটি আটানব্বই লক্ষ চার হাজার" - This is Bengali, not Punjabi
            
            # # Proper Punjabi test cases
            # "ਨੌਂ ਸੌ ਸਤਾਵਨ ਕਰੋੜ ਅਠਾਨਵੇਂ ਲੱਖ ਚਾਰ ਹਜ਼ਾਰ",  # Large number test
            # "ਇੱਕ ਦੋ ਤਿੰਨ ਚਾਰ ਪੰਜ ਛੇ ਸੱਤ ਅੱਠ ਨੌ ਸਿਫਰ",  # Digits
            # "ਸੱਤ ਸੌ ਸੱਤਾਈ",  # 727
            # "ਤਿੰਨ ਕਰੋੜ ਪੰਜ ਲੱਖ",  # 3,05,00,000
        ]
        
        print("Running Inference on Punjabi Test Texts...")
        print("=" * 60)
        for text in test_texts:
            try:
                # Run Inference
                itn_output = normalizer.inverse_normalize(text, verbose=False)
                print(f"Original : {text}")
                print(f"ITN Output: {itn_output}")
                print("-" * 40)
            except Exception as e:
                print(f"Error processing '{text}': {e}")
                print("-" * 40)
                
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    time_start = time.time()
    run_panjabi_itn()
    time_end = time.time()
    print(f"Total Inference Time: {time_end - time_start:.2f} seconds")
