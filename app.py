from flask import Flask, render_template

app = Flask(__name__)

#@app.route('/')
@app.route('/')
def index():
    #return'<div class="center-div"><img src="map.png" class="displayed" /></div><br /><div class="gitHub"><a href="http://www.google.fr" target="_blank"> Open-source code </a><img src="src/githubj.png" width="30px" /></div>'
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)