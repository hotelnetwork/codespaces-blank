@app.route('/predict/<symbol>')
def predict(symbol):
    fig_json, rmse = predict_stock(symbol)
    return render_template('predict.html', plot=fig_json, rmse=rmse)


@app.route('/stream')
def stream():
    def eventStream():
        while True:
            yield sse.send({"message": "Hello, World!"}, type='greeting')
            return Response(eventStream(), mimetype="text/event-stream")


if __name__ == '__main__':
    app.run(debug=True)
