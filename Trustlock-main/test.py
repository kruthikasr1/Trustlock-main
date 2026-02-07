import joblib

def detect_spamemail(email_text):
    model = joblib.load('model_outputs/model/multinomial_nb_spam_detector.pkl')
    prediction = model.predict([email_text])[0]
    if prediction == 0:
        return "This is a Normal Email!"
    else:
        return "This is a Spam Email!"

#print(detect_spam('Free Tickets for IPL'))