import base64
import os
import dash
from dash import dcc, html, Input, Output, State, ctx
from utils.classify_image import classify_image
from utils.youtube_sentiment import get_trending_videos, fetch_arabic_comments, analyze_comments

app = dash.Dash(__name__)
server = app.server
app.title = "تحليل مشاعر الصور والتعليقات"

# 🔐 مفتاح YouTube API من المتغير البيئي
YOUTUBE_API_KEY = os.getenv("AIzaSyANEG0NbdmV_veIiZHY9cyK-0du_cYmtRk")

app.layout = html.Div([
    html.H1("📊 مشروع تحليل المشاعر - هنودي تاج راسي 👑", style={'textAlign': 'center'}),

    html.H2("📷 تحليل صورة"),
    dcc.Upload(
        id='upload-image',
        children=html.Button('ارفع صورة'),
        multiple=False
    ),
    html.Div(id='image-result', style={'marginTop': '20px'}),

    html.Hr(),

    html.H2("🎥 تحليل تعليقات فيديو ترند حسب موضوع"),
    dcc.Input(id='topic-input', type='text', placeholder='اكتب موضوع مثل: كرة قدم', style={'width': '50%'}),
    html.Button('🔍 بحث عن الفيديوهات', id='search-button', n_clicks=0),
    dcc.Dropdown(id='video-dropdown', placeholder='اختر فيديو'),
    html.Button('✅ صنف التعليقات', id='analyze-comments', n_clicks=0),
    html.Div(id='comments-result', style={'marginTop': '20px'})
])

# 📸 تحليل صورة
@app.callback(
    Output('image-result', 'children'),
    Input('upload-image', 'contents'),
    State('upload-image', 'filename')
)
def process_image(content, filename):
    if content is None:
        return ''
    _, b64 = content.split(',')
    path = f"temp_{filename}"
    with open(path, 'wb') as f:
        f.write(base64.b64decode(b64))
    label, probs = classify_image(path)
    os.remove(path)
    return html.Div([
        html.Img(src=content, style={'width': '300px'}),
        html.H4(f"📷 النتيجة: {label}")
    ])

# 🔍 جلب فيديوهات ترند
@app.callback(
    Output('video-dropdown', 'options'),
    Input('search-button', 'n_clicks'),
    State('topic-input', 'value')
)
def fetch_videos(n, topic):
    if n == 0 or not topic:
        return []
    results = get_trending_videos(api_key=YOUTUBE_API_KEY, query=topic, region='SA')
    return [{'label': title, 'value': vid} for vid, title in results]

# 💬 تحليل تعليقات الفيديو
@app.callback(
    Output('comments-result', 'children'),
    Input('analyze-comments', 'n_clicks'),
    State('video-dropdown', 'value')
)
def analyze(n, video_id):
    if n == 0 or not video_id:
        return ''
    comments = fetch_arabic_comments(video_id)
    results, summary = analyze_comments(comments)
    return html.Div([
        html.H4("📊 ملخص المشاعر:"),
        html.Ul([html.Li(f"{label}: {count} تعليق") for label, count in summary.items()]),
        html.Hr(),
        html.H4("📝 التفاصيل:"),
        html.Ul([
            html.Li([
                html.Span(f"🗣️ {text} ", style={'fontWeight': 'bold'}),
                html.Span(f"→ التصنيف: {sentiment}")
            ]) for text, sentiment, _ in results
        ])
    ])

if __name__ == '__main__':
    app.run_server(debug=True)
