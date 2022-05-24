import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import warnings
data = pd.read_xlsx("Book1.xlsx")
df = data.pivot_table(index="Facilities", columns="Email", values="Rating").fillna(0)
df_matrix = csr_matrix(df.values) # converting the table to an array matrix
model_knn = NearestNeighbors(metric="cosine", algorithm="brute")
model_knn.fit(df_matrix)

import pickle
pickle.dump(model_knn, open('modelknn.pkl','wb'))
warnings.filterwarnings("ignore")

import pickle
model = pickle.load(open("modelknn.pkl","rb")) #loading the created model

st.set_page_config(page_title="CANCER Application") #tab title

#prediction function
def predict_status(Email,Facilities,Rating):
    input_data = np.asarray([Email,Facilities,Rating])
    input_data = input_data.reshape(1,-1)
    prediction = model.predict(input_data)
    return prediction[0]

def main():

    # titling your page
    st.title("Product Recommendation System")
    st.write("A quick ML app to Bank Product Recommendation ")

    #getting the input
    Email= st.text_input("Enter your Email")
   



    #predict value
    diagnosis = ""

    if st.button("Recommendation are here"):
        diagnosis = predict_status(Email,Facilities,Rating)
        if diagnosis=="1":
            st.info("You Have an Recommendation")
            #st.markdown("![You're like this!](https://i.gifer.com/L6m.gif)")
        elif diagnosis=="0":
                st.info(" not Recommendation Something Else")
                #st.markdown("![You're like this!](https://i.gifer.com/L6m.gif)")

        
        else:
                st.error("noooo")

            
    
        
        
        
        st.write("## Thank you for Visiting \nProject by Suyog H")
        #st.markdown("<h1 style='text-align: right; color: blue; font-size: small;'><a href='https://github.com/suyog56/CANCER'>Looking for Source Code?</a></h1>", unsafe_allow_html=True)
        # st.markdown("<h1 style='text-align: right; color: white; font-size: small'>you can find it on my GitHub</h1>", unsafe_allow_html=True)

    


if __name__=="__main__":
    main()