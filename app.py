from flask import Flask, request, render_template
import numpy as np
from data_clean import main
import data_clean
import matplotlib.pyplot as plt
from io import BytesIO
import base64
###This program is used for initialize and estabilsh the webpages useing FLASK frame work
###This is also the main program to run to start interacting with the project
app = Flask(__name__) ###initialize the flask app

# define 3 questions that will be asked in the home page
questions = ['Do you need to live alone? [yes / no]', 'What is your acceptable number of shared rooms? For example, [1-3 bed]?', "What's your acceptable price? For example, [1,000-3,000]"]

# define the defualt url content
@app.route('/', methods=['GET', 'POST'])
def index():
    ''' Define the homepage of the project, Ask three questions to the user that helped with recommendation 
    When open the webpage for the first time(GET request), the page would be directed to 'index.html', when user finished answering the
    questions and click submit, the webpaged will be directied to recommmendation result page (POST request, to 'table.html')
     
    Parameters
    ----------
    if_studio: string
        The answer to Do you need to live alone? [yes / no]
        
    beds: string
        The answer to What is your acceptable number of shared rooms? For example, [1-3 bed]
    
    price: string    
        The answer to What's your acceptable price? For example, [1,000-3,000] 
        
    data: DataFrame
        The recommendation result based on users' anwsaers

    Returns
    -------
    table.html
        The webpage used for displaying recommendation results
        
    index.html
        The homepage
    '''    
    # process users' answer if is POST request
    if request.method == 'POST':
        if_studio = request.form['answer1']
        beds = request.form['answer2']
        price = request.form['answer3']

        d = main(price, beds, if_studio)
        global data
        data = d.iloc[:10, :-1]
        data['url'] = data['url'].apply(lambda x: '<a href="{0}">{0}</a>'.format(x))

        return render_template('table.html', df=data)
    # display questions if GET request
    else:
        return render_template('index.html', questions=questions)

@app.route('/details/<id>')
def details(id):
    ''' Define the detailed information page of the project. 
    Displaying one specific apartment information  including title, address, price, URL, floor plan. 
    A line chart and a bar chart is also displayed to show the min and mix price of the 10 recommnedations
    in the table page. 
     
    Parameters
    ----------     
    row/row2: DataFrame
        The one specific recommendation result.
     
    df/data: DataFrame
        10 recommendation results from the last page
        
    details: string
        The "headline" for the detailed page
        
    min_: int
        The minimum price of the one specific recommendation result
        
    max_: int
        The maximum price of the one specific recommendation result
        
    plot_url: Base64 string
        The base64 string for the line chart
        
    plot_url2: Base64 string
        The base64 string for the bar chart        

    Returns
    -------
    detail.html
        The webpage used for displaying one detailed apartment information
    '''
    # Select the row of the specific apartment information
    row = data.loc[data['title']==id]

    details = f'Details for item with {id}'

    # data cleansing
    df = data.copy()
    df = df[['pricing','title']]
    df['pricing'] = df['pricing'].apply(data_clean.get_price)
    df['min_price'] = df['pricing'].str[0].astype(int)
    df['max_price'] = df['pricing'].str[1].astype(int)
    df = df.drop('pricing',axis=1)

    # process the price data
    row2 = row.copy()
    row2 = row2[['pricing', 'title']]
    row2['pricing'] = row2['pricing'].apply(data_clean.get_price)
    row2['min_price'] = row2['pricing'].str[0].astype(int)
    row2['max_price'] = row2['pricing'].str[1].astype(int)
    max_ = row2.iloc[0,-1]
    min_ = row2.iloc[0,-2]

    # plot the line chart
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    df.plot(ax=ax)
    plt.xticks(range(0, 100, 10), df['title'].str[:20], rotation=15)
    plt.axhline(max_,linestyle='-.',color='r')  #red dash line indicating the max price of this specific search result
    plt.axhline(min_,linestyle='-.',color='r')
    # save the plot in PNG format and convert it into Base64 string
    img = BytesIO()
    fig.savefig(img)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()


    # plot the bar chart
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    name = df['title'].values
    max = df['max_price'].values
    min = df['min_price'].values
    x = np.arange(len(name))
    plt.xticks(x,name,rotation=15)
    ax.bar(x+0.1,max,width=0.2,color='#008B8B')
    ax.bar(x-0.1,min,width=0.2,color='#002B2B')
    img = BytesIO()
    fig.savefig(img)
    img.seek(0)
    plot_url2 = base64.b64encode(img.getvalue()).decode()

    return render_template('details.html', details=details,df=row,plot_url=plot_url,plot_url2=plot_url2)

if __name__ == '__main__':
    app.run(debug=False)