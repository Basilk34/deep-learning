


    query = st.text_input("🔍 ابحث داخل المنشورات:")
    if query:
        results = [p for p in posts if query.lower() in p.lower()]
        if results:
            st.write("### ✅ النتائج:")
            for res in results:
                st.write(f"- {res}")
                seq = tokenizer.texts_to_sequences([res])
                padded = pad_sequences(seq, maxlen=10)
                pred = model.predict(padded, verbose=0)
                label = label_map[np.argmax(pred)]
                st.write(f"🧠 التوجه: `{label}`")
        else:
            st.warning("🚫 لا توجد نتائج مطابقة.")
