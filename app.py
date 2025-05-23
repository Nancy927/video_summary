#app.py
from flask import Flask, render_template # type: ignore
from routes.video_summary import video_summary_api
from flask_cors import CORS  # type: ignore
app = Flask(__name__, static_folder='static', template_folder='templates')
app.register_blueprint(video_summary_api, url_prefix='/api')  # 注册蓝图
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 限制为100MB
CORS(app)


@app.route('/')
def index():
    return render_template("index.html")  # 渲染 HTML 页面

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # 启动 Flask 服务

