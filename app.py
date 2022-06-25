from numpy import size
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from plotly import graph_objs as go
from sklearn.linear_model import LinearRegression
import numpy as np

df = pd.read_csv('Salary_Data.csv')

lr = LinearRegression()
x = np.array(df['YearsExperience']).reshape(-1,1)
y = np.array(df['Salary']).reshape(-1,1)
lr.fit(x,y)

st.title('Salary Predictor')
st.image('images.jpg',width=800)

nav = st.sidebar.radio('Navigation',['Home','Predict','Contribute'])

if nav=='Home':

    if st.checkbox('Show table'):
        st.table(df)

    graph = st.selectbox('Type of Graph',['Interactive','Non-interactive'])

    val = st.slider('Minimum Experience',0,20)
    df = df.loc[df['YearsExperience']>=val]

    if graph=='Interactive':
        trace = go.Scatter(x=df['YearsExperience'],y=df['Salary'],mode='markers')
        data = [trace]
        layout = go.Layout(title='Salary Trends with Experience',
                            xaxis={'title':'Experience'},
                            yaxis={'title':'Salary'})
        
        fig = go.Figure(data=data,layout=layout)
        st.plotly_chart(fig)

    elif graph=='Non-interactive':
        fig,ax = plt.subplots()
        ax.scatter(df['YearsExperience'],df['Salary'])
        ax.set_xlabel('Experience')
        ax.set_ylabel('Salary')

        st.pyplot(fig)

elif nav=='Predict':
    
    st.header('Know Your Salary')
    val = st.number_input('Enter your experience',0.00,20.00,step=0.25)
    val = np.array(val).reshape(1,-1)
    pred = lr.predict(val)[0]

    if st.button('Predict'):
        st.success('Your predicted salary is {}'.format(pred))

elif nav=='Contribute':
    
    st.header('Contribute to our Dataset')

    ex = st.number_input('Enter your Experience',0.0,20.0,step=0.25)
    sal = st.number_input('Enter your Salary',0.00,1000000.00,step=1000.00)

    if st.button('Submit'):
        to_add = {"YearsExperience":ex,
                "Salary":sal}
        to_add = pd.DataFrame(to_add)
        to_add.to_csv('Salary_Data.csv',mode='a',header=False,index=True)
        st.success('Data Submitted')