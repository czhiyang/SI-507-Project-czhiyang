The README containing any special instructions for running your code (e.g., how to supply API keys) as well as a brief description of how to interact with your program.

How to interact:
    1.To run the main project and interact with the Ann Arbor apartment searching system: run "app.py"
    2.To verify/update data source from Crawling and scraping from multiple websites: run "down_data.py"
    3.To process the data and construct the data sturcture: run "data_process_and_tree_construction.py.py"
    4.To read the decision tree that is from the project: run "Read_Tree.py"

Special instructions:
    1.To successfully run the function of data crawling and scraping a chromedriver needs to be added into both  python and proejct file.
    Please download the drive according to the chrome version you are currently using:
    https://registry.npmmirror.com/binary.html?path=chromedriver/
    2.To see the constructed tree in picture, a software graphviz is needed to be installed.
    Please visit the website below to download:
    https://graphviz.org/download/

Required python packages:
    Flask, pandas, io, base64, sklearn, graphviz, selenium, lxml, request
