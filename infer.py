from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer

def run_marathi_itn():
    # Initialize the InverseNormalizer for Marathi ('mr')
    # Note: Setting verbose=False suppresses debug logs
    try:
        print("Loading Inverse Normalizer for Bengali...")
        normalizer = InverseNormalizer(lang='bn')
        
        # Example Bengali texts (Spoken -> Written format)
        # "দুই হাজার চব্বিশ" -> 2024
        # "শত টাকা" -> 100 ₹
        test_texts = [
            "নয় নয়শো সাতান্ন কোটি আটানব্বই লক্ষ চার হাজার",
            "চারটা পঁয়ত্রিশে খেতে গিয়েছিলাম। পৌনে পাঁচটার মধ্যে যেতে হবে",
            "আজকের তারিখ আঠারো মার্চ দুই হাজার ছাব্বিশ। মানে আগামীকাল উনিশ মার্চ হবে। অথবা কেউ একে মার্চ আঠারো দুই হাজার ছাব্বিশও বলতে পারে",
            "আমার আধার কার্ড নম্বর একষট্টি চুয়াল্লিশ চুয়াত্তর সাতাশ দুই হাজার একশো সাতাত্তর এবং মোবাইল নম্বর এক দুই তিন চার পাঁচ ছয় সাত আট নয় শূন্য",
            "আমার আধার কার্ড নম্বর সিক্সটি ফোরটি ফোর সেভেন্টি টুয়েন্টি সেভেন টু থাউজ্যান্ড ওয়ান হান্ড্রেড অ্যান্ড সেভেন্টি এইট এবং মোবাইল নম্বর ওয়ান টু থ্রি ফোর ফাইভ সিক্স সেভেন এইট নাইন জিরো",
            "আমার আধার কার্ড নম্বর ষাট পঁয়তাল্লিশ সত্তর সাতাশ দুই হাজার একশো আটাত্তর এবং মোবাইল নম্বর এক দুই তিন চার পাঁচ ছয় সাত আট নয় শূন্য",
        ]
        
        for text in test_texts:
            # Run Inference
            itn_output = normalizer.inverse_normalize(text, verbose=False)
            print("-" * 40)
            print(f"Original : {text}")
            print(f"ITN Output: {itn_output}")  
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_marathi_itn()
