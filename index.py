from jinx import tkr, concurrent, os, keras, itl, data, math, pd, plt, np, tf, sns, datetime, y, w, F, render_template, HTTPBasicAuth, generate_password_hash, check_password_hash, linear_model, r2_score, mean_squared_error, px, go, request, redirect_stdout, io, jsonify, beta, norm, binom, poisson, laplace, uniform, expon, gamma, t, chi2, f, multivariate_normal, train_data, test_data, train_labels, test_labels


# from jinja2 import Template

app = F(__name__)

auth = HTTPBasicAuth()


@app.route('/', methods=['GET'])
def home():
    symbol = tkr
    # Load the data from yfinance of the 1m interval of the last 7 days
    # data = y.download(symbol, period='7d', interval='1m')

    # Convert the data to a pandas dataframe in reverse order
    df = pd.DataFrame(data).iloc[::-1]

    # Convert the dataframe to HTML and return it
    # return df.to_html() with the dataframe as the context and symbol as the symbol

    # Get last time and date the file was accessed using Month names and 12 hours w/ AM & PM in Pacific Standard Time.
    last_updated = datetime.now().strftime("%B %d, %Y at %I:%M %p")

    return render_template(
        'index.html',
        df=df, symbol=symbol.capitalize(), last_updated=last_updated)

# Define a function to get the model summary
# Use a Template to render the dataframe as HTML


def get_model_summary():

    buffer = io.StringIO()
    model.summary(print_fn=lambda x: buffer.write(x + '\n'))
    return model

# Define a route for the model page


@app.route('/model')
def model():
    # Get the model summary
    model_summary = get_model_summary()

    # Render the model.html template and pass in the model summary
    return render_template('model.html', model_summary=model_summary)

# Define a route for the matplotlib page


@app.route('/matplotlib')
def matplotlib():

    df = pd.DataFrame(data)
    # Calculate the moving average
    df['MA'] = df['Close'].rolling(window=20).mean()
    df['STDEV'] = df['Close'].rolling(window=20).std()
    df['Upper'] = df['MA'] + (df['STDEV'] * 2)
    df['Lower'] = df['MA'] - (df['STDEV'] * 2)
    df['MAFast'] = df['Close'].rolling(window=5).mean()
    df['MASlow'] = df['Close'].rolling(window=50).mean()
    df['MA'] = df['MA'].fillna(0)
    # ... existing code ...

    # Calculate the stdev of the days the maslow is under the mafast
    df['stdvmaslow'] = np.where(
        df['MASlow'] < df['MAFast'], df['Close'].rolling(window=20).std(), 0)
    # Reverse order of stdvmaslow
    df['stdvmaslow'] = df['stdvmaslow'][::-1]
    # Decide Buy or sell based on the stdev of the days the maslow is under the mafast
    df['Buy'] = np.where(df['stdvmaslow'] > 0, 'Buy', 'Sell')

    # Stdv of first 15 minutes then next hour and 45, then the next 2.5 hours then the next hour and 45 minutes, the the last 15 minutes of the day.
    # Using the 1 minute interval, also the 5 minute interval, the 15 minute interval.

    # Convert the dataframe to a numpy array
    jinx = df.to_numpy()

    # Create Timer
    start_time = datetime.now()

    # plt = sns.lineplot(data=X)
    # Add the moving average to the figure
    fig = go.Figure(data=[
        go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                       low=df['Low'], close=df['Close'], name='market data'),
        go.Line(x=df.index, y=df['MAFast'], mode='lines',
                name='MAFast', line=dict(color='teal', width=1)),
        go.Line(x=df.index, y=df['MA'], mode='lines', name='MA', line=dict(color='black', width=1)),
        go.Line(x=df.index, y=df['MASlow'], mode='lines',
                name='MASlow', line=dict(color='orange', width=1)),
        go.Line(x=df.index, y=df['Upper'], mode='lines',
                name='Upper', line=dict(color='green', width=1)),
        go.Line(x=df.index, y=df['Lower'], mode='lines',
                name='Lower', line=dict(color='red', width=1))
    ],
        layout=go.Layout(title=go.layout.Title(
            text=tkr + " Candlestick Chart"))
    )
    fig.update_layout(xaxis_rangeslider_visible=True, xaxis_rangeslider_thickness=0.05)
    fig.update_layout(
        xaxis_title='Date', yaxis_title='Price',
        font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"
                  )
    )
    # Add Buttons and Annotations
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=list([
                    dict(
                        args=[{'visible': [True, True, True, True, True, True]}],
                        label="All",
                        method="update"
                    ),
                    dict(
                        args=[{'visible': [True, False, False, False, False, False]}],
                        label="Candlestick",
                        method="update"
                    ),
                    dict(
                        args=[{'visible': [False, True, False, False, False, False]}],
                        label="MA",
                        method="update"
                    ),
                    dict(
                        args=[{'visible': [False, False, True, False, False, False]}],
                        label="MAFast",
                        method="update"
                    ),
                    dict(
                        args=[{'visible': [False, False, False, True, False, False]}],
                        label="MASlow",
                        method="update"
                    ),
                    dict(
                        args=[{'visible': [False, False, False, False, True, True]}],
                        label="Bollinger Bands",
                        method="update"
                    )
                ]),
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.1,
                yanchor="top"
            ),
        ]
    )
    # Add Annotations
    fig.update_layout(annotations=[
        dict(x='2022-01-01', y=0, xref="x", yref="y", text="2022",
             showarrow=True, arrowhead=7, ax=0, ay=-40)
    ])
    # Zoom y-axis and auto-scale
    fig.update_yaxes(fixedrange=False)
    # Add Legend
    fig.update_layout(
        legend=dict(x=0, y=1, traceorder="normal",
                    font=dict(family="sans-serif", size=12, color="black"),
                    bgcolor="LightSteelBlue", bordercolor="Black", borderwidth=2
                    )
    )
    # Add Annotations
    fig.update_layout(annotations=[
        dict(x='2022-01-01', y=0, xref="x", yref="y", text="2022",
             showarrow=True, arrowhead=7, ax=0, ay=-40)
    ])

    # Not in pop-up
    # fig.show()
    fig.write_html("templates/candlestick.html")
    # fig.show()
    # Render the matplotlib.html template
    return render_template('matplotlib.html', fig=fig, jinx=jinx, df=df)

# Define the route for the search_autocomplete page


@app.route('/search')
def search():
    if request.method == 'POST':
        tkr = request.form.get('ticker', '')  # Get the 'ticker' field from the form
        itl = request.form.get('interval', '')  # Get the 'interval' field from the form
        # Now you can use tkr and itl for further processing
        # ...
        # Render the template of the page that called the route.
        return jsonify({'status': 'success', 'message': 'Form submitted'})

# Define Route for Bayes


@app.route('/bayes')
def bayes():
    # Load the data from file
    train_data = np.load('train_data.npy')
    test_data = np.load('test_data.npy')
    train_labels = np.load('train_labels.npy')
    test_labels = np.load('test_labels.npy')

    # predict using Bayesian Ridge Regression
    try:
        # Print the shapes of the data
        # print(train_data.shape, test_data.shape, train_labels.shape, test_labels.shape)

        # Reshape the data to 2D if it is 3D
        if len(train_data.shape) == 3:
            train_data = train_data.reshape(train_data.shape[0], -1)

        # Create a Bayesian Ridge Regression model with default parameters
        clf = linear_model.BayesianRidge(
            n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06)

        # Fit the model on the training data
        clf.fit(train_data, train_labels)

        # Reshape the data to 2D if it is 3D
        if len(test_data.shape) == 3:
            test_data = test_data.reshape(test_data.shape[0], -1)

        # Use the model to predict on the test data
        predictions = clf.predict(test_data)
        # Slice the predictions to only the last 100
        predictions = predictions[-100:]
        # Reverse order of predictions
        predictions = predictions[::-1]

        # Slice the test labels to only the last 100
        test_labels = test_labels[-100:]
        # Reverse order of test labels
        test_labels = test_labels[::-1]

        # Calculate the mean squared error
        mse = mean_squared_error(test_labels, predictions)

        # Print the mean squared error
        print("Mean squared error:", mse)

        # Calculate the coefficient of determination (R^2)
        r2 = r2_score(test_labels, predictions)

        # Calculate Linear Regression
        linear = linear_model.LinearRegression()
        linear.fit(train_data, train_labels)
        lpredictions = linear.predict(test_data)
        # Slice the predictions to only the last 100
        lpredictions = lpredictions[-100:]
        # Reverse order of lpredictions
        lpredictions = lpredictions[::-1]

        # LSTM data

        # Print the coefficient of determination # Print the classification report # Print the confusion matrix # Print the accuracy score # Print the analysis # Print the predictions # Print the true labels
        print("Coefficient of determination:", r2)
        print("Data:", df.tail(10))
        print("Predictions:", predictions[-10:].tolist())
        print("Linear Predictions:", lpredictions[-10:].tolist())
        print("True labels:", test_labels[-10:].tolist())

        fig = go.Figure(
            data=[
                go.Line(x=predictions.index, y=predictions,
                        mode='lines', name='Predictions'),
                go.Line(x=predictions.index, y=test_labels,
                        mode='lines', name='True Profit')
            ], layout=go.Layout(title=go.layout.Title(text=tkr + " Predictions"))
        )
        fig.update_layout(yaxis_title='Predictions')
        fig.update_xaxes(title_text='Jinx')
        fig.write_html("templates/bayes_plot.html")

        # Create a plot
        plt.figure(figsize=(10, 5))
        plt.plot(test_labels, label='True')
        plt.plot(predictions, label='Predicted')
        plt.plot(lpredictions, label='Linear Predicted')
        plt.legend()

        # Save the plot as an image
        # ' + str(datetime.now()) + '
        image_path = 'static/images/bayes_plt' + str(datetime.now()) + '.png'
        plt.savefig(image_path)

        fig01 = go.Figure(
            data=[
                go.Line(x=predictions.index, y=lpredictions,
                        mode='lines', name='Linaer Predictions'),
            ],
            layout=go.Layout(title=go.layout.Title(
                text=tkr + " Linear Predictions"))
        )
        fig01.update_layout(yaxis_title='Linear Predictions')
        # Create linear_regression_plot.html if it does not exist, write
        # either way.
        fig01.write_html("linear_regression_plot" + str(datetime.now()) + ".html")

        # Create a plot
        plt.figure(figsize=(10, 5))
        plt.plot(test_labels, label='True')
        plt.plot(lpredictions, label='Linaer Predicted')
        plt.legend()

        # Save the plot as an image
        # tkr +
        img_path = 'static/images/linear_regression_plot' + tkr + str(datetime.now()) + '.png'
        plt.savefig(img_path)

    except:
        print("Error: Data not found.")

    # Render the bayesian_ridge.html template predictions will only be last 100
    # Format predictions to 2 decimal places.
    # , analysis=analysis, classification_report=classification_report, confusion_matrix=confusion_matrix, accuracy_score=accuracy_score, image_path=image_path)
    return render_template('bayes.html', lpredictions=lpredictions.round(2), predictions=predictions.round(2), train_data=train_data, test_data=test_data, train_labels=train_labels, test_labels=test_labels, clf=clf, r2=r2, mse=mse)

# Predict the next three days using the Bayesian Ridge Regression model
# Define Route for Bayesian Ridge Regression


@app.route('/bayesian_ridge')
def bayesian_ridge():
    # Load the data from file
    train_data = np.load('train_data.npy')
    test_data = np.load('test_data.npy')
    train_labels = np.load('train_labels.npy')
    test_labels = np.load('test_labels.npy')

    # Print the shapes of the data
    print(train_data.shape, test_data.shape, train_labels.shape, test_labels.shape)

    # Reshape the data to 2D if it is 3D
    if len(train_data.shape) == 3:
        train_data = train_data.reshape(train_data.shape[0], -1)

    # Create a Bayesian Ridge Regression model with default parameters
    clf = linear_model.BayesianRidge()

    # Fit the model on the training data
    clf.fit(train_data, train_labels)

    # Reshape the data to 2D if it is 3D
    if len(test_data.shape) == 3:
        test_data = test_data.reshape(test_data.shape[0], -1)

    # Use the model to predict on the test data
    predictions = clf.predict(test_data)

    # Print the coefficient of determination
    print("Coefficient of determination:", r2_score(test_labels, predictions))

    # Create a plot
    plt.figure(figsize=(10, 5))
    plt.plot(test_labels, label='True')
    plt.plot(predictions, label='Predicted')
    plt.legend()

    # Save the plot as an image
    plt.savefig('static/images/bayesian_ridge_plot.png')

    # Render the bayesian_ridge.html template
    return render_template('bayesian_ridge.html', r2_score=r2_score, test_labels=test_labels, predictions=predictions, clf=clf)

# Define Route for Linear Regression
# @app.route('/<jinx_function>', methods=['GET'], defaults={'jinx_function': 'linear_regression', 'jinx_function': 'random_forest', 'jinx_function': 'support_vector_machine', 'jinx_function': 'decision_tree', 'jinx_function': 'logistic_regression', 'jinx_function': 'neural_network', 'jinx_function': 'gradient_boosting', 'jinx_function': 'k_nearest_neighbors', 'jinx_function': 'xgboost', 'jinx_function': 'lightgbm', 'jinx_function': 'catboost', 'jinx_function': 'automl', 'jinx_function': 'time_series', 'jinx_function': 'clustering', 'jinx_function': 'arima', 'jinx_function': 'prophet', 'jinx_function': 'prophet_forecast', 'jinx_function': 'prophet_forecast_plot', 'jinx_function': 'prophet_forecast_plot_interactive', 'jinx_function': 'prophet_forecast_plot_interactive_plotly'})
# def calculations(jinx_function):
#     # Render the linear_regression.html template
#     return render_template('<jinx_function>.html')


# Run the app on the local development server

# add new page
@app.route('/linear_regression')
def linear_regression():
    # Load the data from file
    train_data = np.load('train_data.npy')
    test_data = np.load('test_data.npy')
    train_labels = np.load('train_labels.npy')
    test_labels = np.load('test_labels.npy')

    # Print the shapes of the data
    print(train_data.shape, test_data.shape, train_labels.shape, test_labels.shape)

    # Reshape the data to 2D if it is 3D
    if len(train_data.shape) == 3:
        train_data = train_data.reshape(train_data.shape[0], -1)

    # Create a Linear Regression model
    clf = linear_model.LinearRegression()

    # Fit the model on the training data
    clf.fit(train_data, train_labels)

    # Reshape the data to 2D if it is 3D
    if len(test_data.shape) == 3:
        test_data = test_data.reshape(test_data.shape[0], -1)

    # Use the model to predict on the test data
    predictions = clf.predict(test_data)

    # Print the coefficient of determination
    print("Coefficient of determination:", r2_score(test_labels, predictions))

    # Generate Intercept and Slope
    intercept = clf.intercept_
    slope = clf.coef_

    # Display resluts in table
    # Create a table
    table = go.Figure(data=[go.Table(
        header=dict(values=['True', 'Predicted'],
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[test_labels, predictions],
                   fill_color='lavender',
                   align='left'))
    ])

    # Create a plot
    plt.figure(figsize=(10, 5))
    plt.plot(test_labels, label='True')
    plt.plot(predictions, label='Predicted')
    plt.annotate('Linear Regression', xy=(0.5, 0.5), xytext=(
        0.5, 0.5), textcoords='axes fraction', arrowprops=dict(facecolor='black', shrink=0.05))
    plt.legend()

    # Save the plot as an image
    plt.savefig('static/images/linear_regression_plot.png')

    fig = go.Figure(
        data=[
            go.Line(y=predictions,
                    mode='lines', name='Predictions'),
            go.Line(y=test_labels,
                    mode='lines', name='True Profit')
        ], layout=go.Layout(title=go.layout.Title(text=tkr + " Predictions"))
    )
    fig.update_layout(yaxis_title='Predictions')
    fig.update_xaxes(title_text='Jinx')
    fig.write_html("templates/linear_regression_plot.html")

    # Render the linear_regression.html template
    return render_template('linear_regression.html', fig=fig, table=table, intercept=intercept, slope=slope, test_labels=test_labels, predictions=predictions)


# add new page
@app.route('/predict')
def predict():
    return render_template('linear_regression_plot.html')


@app.route('/random_forest')
def random_forest():
    # Load the data from file
    train_data = np.load('train_data.npy')
    test_data = np.load('test_data.npy')
    train_labels = np.load('train_labels.npy')
    test_labels = np.load('test_labels.npy')

    # Print the shapes of the data
    print(train_data.shape, test_data.shape, train_labels.shape, test_labels.shape)

    # Reshape the data to 2D if it is 3D
    if len(train_data.shape) == 3:
        train_data = train_data.reshape(train_data.shape[0], -1)

    # Import the Random Forest model
    from sklearn import ensemble

    # Create a Random Forest model
    clf = ensemble.RandomForestRegressor()

    # Fit the model on the training data
    clf.fit(train_data, train_labels)

    # Reshape the data to 2D if it is 3D
    if len(test_data.shape) == 3:
        test_data = test_data.reshape(test_data.shape[0], -1)

    # Use the model to predict on the test data
    predictions = clf.predict(test_data)

    # Print the coefficient of determination
    print("Coefficient of determination:", r2_score(test_labels, predictions))

    # Create a plot
    plt.figure(figsize=(10, 5))
    plt.plot(test_labels, label='True')
    plt.plot(predictions, label='Predicted')
    plt.legend()

    # Save the plot as an image
    plt.savefig('static/images/random_forest_plot.png')

    # Render the random_forest.html template
    return render_template('random_forest.html')

# add new page


@app.route('/support_vector_machine')
def support_vector_machine():
    # Load the data from file
    train_data = np.load('train_data.npy')
    test_data = np.load('test_data.npy')
    train_labels = np.load('train_labels.npy')
    test_labels = np.load('test_labels.npy')

    # Print the shapes of the data
    print(train_data.shape, test_data.shape, train_labels.shape, test_labels.shape)

    # Reshape the data to 2D if it is 3D
    if len(train_data.shape) == 3:
        train_data = train_data.reshape(train_data.shape[0], -1)

    # Create a Support Vector Machine model
    clf = svm.SVR()

    # Fit the model on the training data
    clf.fit(train_data, train_labels)

    # Reshape the data to 2D if it is 3D
    if len(test_data.shape) == 3:
        test_data = test_data.reshape(test_data.shape[0], -1)

    # Use the model to predict on the test data
    predictions = clf.predict(test_data)

    # Print the coefficient of determination
    print("Coefficient of determination:", r2_score(test_labels, predictions))

    # Create a plot
    plt.figure(figsize=(10, 5))
    plt.plot(test_labels, label='True')
    plt.plot(predictions, label='Predicted')
    plt.legend()

    # Save the plot as an image
    plt.savefig('static/images/support_vector_machine_plot.png')

    # Render the support_vector_machine.html template
    return render_template('support_vector_machine.html')

# Add Page: Decision Tree


@app.route('/decision_tree')
def decision_tree():
    # Load the data from file
    train_data = np.load('train_data.npy')
    test_data = np.load('test_data.npy')
    train_labels = np.load('train_labels.npy')
    test_labels = np.load('test_labels.npy')

    # Print the shapes of the data
    print(train_data.shape, test_data.shape, train_labels.shape, test_labels.shape)

    # Reshape the data to 2D if it is 3D
    if len(train_data.shape) == 3:
        train_data = train_data.reshape(train_data.shape[0], -1)

    # Create a Decision Tree model
    clf = tree.DecisionTreeRegressor()

    # Fit the model on the training data
    clf.fit(train_data, train_labels)

    # Reshape the data to 2D if it is 3D
    if len(test_data.shape) == 3:
        test_data = test_data.reshape(test_data.shape[0], -1)

    # Use the model to predict on the test data
    predictions = clf.predict(test_data)

    # Print the coefficient of determination
    print("Coefficient of determination:", r2_score(test_labels, predictions))

    # Create a plot
    plt.figure(figsize=(10, 5))
    plt.plot(test_labels, label='True')
    plt.plot(predictions, label='Predicted')
    plt.legend()

    # Save the plot as an image
    plt.savefig('static/images/decision_tree_plot.png')

    # Render the decision_tree.html template
    return render_template('decision_tree.html')

# Add Page: Logistic Regression


@app.route('/logistic_regression')
def logistic_regression():
    # Load the data from file
    train_data = np.load('train_data.npy')
    test_data = np.load('test_data.npy')
    train_labels = np.load('train_labels.npy')
    test_labels = np.load('test_labels.npy')

    # Print the shapes of the data
    print(train_data.shape, test_data.shape, train_labels.shape, test_labels.shape)

    # Reshape the data to 2D if it is 3D
    if len(train_data.shape) == 3:
        train_data = train_data.reshape(train_data.shape[0], -1)

    # Create a Logistic Regression model
    clf = linear_model.LogisticRegression()

    # Fit the model on the training data
    clf.fit(train_data, train_labels)

    # Reshape the data to 2D if it is 3D
    if len(test_data.shape) == 3:
        test_data = test_data.reshape(test_data.shape[0], -1)

    # Use the model to predict on the test data
    predictions = clf.predict(test_data)

    # Print the classification report
    print("Classification report:", classification_report(
        test_labels, predictions))

    # Print the confusion matrix
    print("Confusion matrix:", confusion_matrix(test_labels, predictions))

    # Print the accuracy score
    print("Accuracy score:", accuracy_score(test_labels, predictions))

    # Render the logistic_regression.html template
    return render_template('logistic_regression.html')

# Add Page: Neural Network


@app.route('/neural_network')
def neural_network():
    # Load the data from file
    train_data = np.load('train_data.npy')
    test_data = np.load('test_data.npy')
    train_labels = np.load('train_labels.npy')
    test_labels = np.load('test_labels.npy')

    # Print the shapes of the data
    print(train_data.shape, test_data.shape, train_labels.shape, test_labels.shape)

    # Reshape the data to fit the model
    x_train = train_data.reshape(train_data.shape[0], train_data.shape[1], 1)
    x_test = test_data.reshape(test_data.shape[0], test_data.shape[1], 1)

    # Define the model
    model = keras.Sequential([
        keras.layers.LSTM(128, input_shape=(train_data.shape[1], 1)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(x_train, train_labels, epochs=100, batch_size=32)

    # Test the model
    model.eval()
    with torch.no_grad():
        inputs = torch.from_numpy(test_data).float()
        targets = torch.from_numpy(test_data).float()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        print(f'Test Loss: {loss.item()}')

    # Render the neural_network.html template
    return render_template('neural_network.html')

# Add Page: Gradient Boosting


@app.route('/gradient_boosting')
def gradient_boosting():
    # Load the data from file
    train_data = np.load('train_data.npy')
    test_data = np.load('test_data.npy')
    train_labels = np.load('train_labels.npy')
    test_labels = np.load('test_labels.npy')

    # Print the shapes of the data
    print(train_data.shape, test_data.shape, train_labels.shape, test_labels.shape)

    # Create a Gradient Boosting model
    clf = ensemble.GradientBoostingRegressor()

    # Fit the model on the training data
    clf.fit(train_data, train_labels)

    # Use the model to predict on the test data
    predictions = clf.predict(test_data)

    # Print the coefficient of determination
    print("Coefficient of determination:", r2_score(test_labels, predictions))

    # Create a plot
    plt.figure(figsize=(10, 5))
    plt.plot(test_labels, label='True')
    plt.plot(predictions, label='Predicted')
    plt.legend()

    # Save the plot as an image
    plt.savefig('static/images/gradient_boosting_plot.png')

    # Render the gradient_boosting.html template
    return render_template('gradient_boosting.html')

# Add Page: K Nearest Neighbors


@app.route('/k_nearest_neighbors')
def k_nearest_neighbors():
    # Load the data from file
    train_data = np.load('train_data.npy')
    test_data = np.load('test_data.npy')
    train_labels = np.load('train_labels.npy')
    test_labels = np.load('test_labels.npy')

    # Print the shapes of the data
    print(train_data.shape, test_data.shape, train_labels.shape, test_labels.shape)

    # Reshape the data to 2D if it is 3D
    if len(train_data.shape) == 3:
        train_data = train_data.reshape(train_data.shape[0], -1)

    # Create a K Nearest Neighbors model
    clf = neighbors.KNeighborsRegressor()

    # Fit the model on the training data
    clf.fit(train_data, train_labels)

    # Reshape the data to 2D if it is 3D
    if len(test_data.shape) == 3:
        test_data = test_data.reshape(test_data.shape[0], -1)

    # Use the model to predict on the test data
    predictions = clf.predict(test_data)

    # Print the coefficient of determination
    print("Coefficient of determination:", r2_score(test_labels, predictions))

    # Create a plot
    plt.figure(figsize=(10, 5))
    plt.plot(test_labels, label='True')
    plt.plot(predictions, label='Predicted')
    plt.legend()

    # Save the plot as an image
    plt.savefig('static/images/k_nearest_neighbors_plot.png')

    # Render the k_nearest_neighbors.html template
    return render_template('k_nearest_neighbors.html')

# Add Page: XGBoost


@app.route('/xgboost')
def xgboost():
    # Load the data from file
    train_data = np.load('train_data.npy')
    test_data = np.load('test_data.npy')
    train_labels = np.load('train_labels.npy')
    test_labels = np.load('test_labels.npy')

    # Print the shapes of the data
    print(train_data.shape, test_data.shape, train_labels.shape, test_labels.shape)

    # Create an XGBoost model
    clf = xgb.XGBRegressor()

    # Fit the model on the training data
    clf.fit(train_data, train_labels)

    # Use the model to predict on the test data
    predictions = clf.predict(test_data)

    # Print the coefficient of determination
    print("Coefficient of determination:", r2_score(test_labels, predictions))

    # Create a plot
    plt.figure(figsize=(10, 5))
    plt.plot(test_labels, label='True')
    plt.plot(predictions, label='Predicted')
    plt.legend()

    # Save the plot as an image
    plt.savefig('static/images/xgboost_plot.png')

    # Render the xgboost.html template
    return render_template('xgboost.html')

# Add Page: LightGBM


@app.route('/lightgbm')
def lightgbm():
    # Load the data from file
    train_data = np.load('train_data.npy')
    test_data = np.load('test_data.npy')
    train_labels = np.load('train_labels.npy')
    test_labels = np.load('test_labels.npy')

    # Print the shapes of the data
    print(train_data.shape, test_data.shape, train_labels.shape, test_labels.shape)

    # Create a LightGBM model
    clf = lgb.LGBMRegressor()

    # Fit the model on the training data
    clf.fit(train_data, train_labels)

    # Use the model to predict on the test data
    predictions = clf.predict(test_data)

    # Print the coefficient of determination
    print("Coefficient of determination:", r2_score(test_labels, predictions))

    # Create a plot
    plt.figure(figsize=(10, 5))
    plt.plot(test_labels, label='True')
    plt.plot(predictions, label='Predicted')
    plt.legend()

    # Save the plot as an image
    plt.savefig('static/images/lightgbm_plot.png')

    # Render the lightgbm.html template
    return render_template('lightgbm.html')

# Add Page: CatBoost


@app.route('/catboost')
def catboost():
    # Load the data from file
    train_data = np.load('train_data.npy')
    test_data = np.load('test_data.npy')
    train_labels = np.load('train_labels.npy')
    test_labels = np.load('test_labels.npy')

    # Print the shapes of the data
    print(train_data.shape, test_data.shape, train_labels.shape, test_labels.shape)

    # Create a CatBoost model
    clf = cb.CatBoostRegressor()

    # Fit the model on the training data
    clf.fit(train_data, train_labels)

    # Use the model to predict on the test data
    predictions = clf.predict(test_data)

    # Print the coefficient of determination
    print("Coefficient of determination:", r2_score(test_labels, predictions))

    # Create a plot
    plt.figure(figsize=(10, 5))
    plt.plot(test_labels, label='True')
    plt.plot(predictions, label='Predicted')
    plt.legend()

    # Save the plot as an image
    plt.savefig('static/images/catboost_plot.png')

    # Render the catboost.html template
    return render_template('catboost.html')

# Add Page: AutoML


@app.route('/automl')
def automl():
    # Load the data from file
    train_data = np.load('train_data.npy')
    test_data = np.load('test_data.npy')
    train_labels = np.load('train_labels.npy')
    test_labels = np.load('test_labels.npy')

    # Print the shapes of the data
    print(train_data.shape, test_data.shape, train_labels.shape, test_labels.shape)

    # Create an AutoML model
    clf = autosklearn.regression.AutoSklearnRegressor()

    # Fit the model on the training data
    clf.fit(train_data, train_labels)

    # Use the model to predict on the test data
    predictions = clf.predict(test_data)

    # Print the coefficient of determination
    print("Coefficient of determination:", r2_score(test_labels, predictions))

    # Create a plot
    plt.figure(figsize=(10, 5))
    plt.plot(test_labels, label='True')
    plt.plot(predictions, label='Predicted')
    plt.legend()

    # Save the plot as an image
    plt.savefig('static/images/automl_plot.png')

    # Render the automl.html template
    return render_template('automl.html')

# Add Page: Time Series


@app.route('/time_series')
def time_series():
    # Load the data from file
    train_data = np.load('train_data.npy')
    test_data = np.load('test_data.npy')
    train_labels = np.load('train_labels.npy')
    test_labels = np.load('test_labels.npy')

    # Convert the data to a pandas dataframe
    df = pd.DataFrame(data)

    # Drop the columns that are not needed
    df = df.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1)

    # Convert the Date column to a datetime object
    df['date'] = pd.to_datetime(df['date'])

    # Set the Date column as the index
    df = df.set_index('date')

    # Print the data
    print(df)

    # Create a plot
    plt.figure(figsize=(10, 5))
    plt.plot(df['Close'], label='Close')
    plt.legend()

    # Save the plot as an image
    plt.savefig('static/images/time_series_plot.png')

    # Render the time_series.html template
    return render_template('time_series.html')

# Add Page: Clustering


@app.route('/clustering')
def clustering():
    # Load the data from file
    data = np.load('data.npy')

    # Print the shape of the data
    print(data.shape)

    # Create a KMeans model
    clf = cluster.KMeans(n_clusters=3)

    # Fit the model on the data
    clf.fit(data)

    # Use the model to predict on the data
    predictions = clf.predict(data)

    # Create a plot
    plt.figure(figsize=(10, 5))
    plt.scatter(data[:, 0], data[:, 1], c=predictions, cmap='viridis')
    plt.legend()

    # Save the plot as an image
    plt.savefig('static/images/clustering_plot.png')

    # Render the clustering.html template
    return render_template('clustering.html')

# Add Page: ARIMA


@app.route('/arima')
def arima():
    # Load the data from file
    data = np.load('data.npy')

    # Print the shape of the data
    print(data.shape)

    # Create an ARIMA model
    clf = ARIMA(data, order=(5, 1, 0))

    # Fit the model on the data
    clf.fit()

    # Use the model to predict on the data
    predictions = clf.predict()

    # Create a plot
    plt.figure(figsize=(10, 5))
    plt.plot(data, label='True')
    plt.plot(predictions, label='Predicted')
    plt.legend()

    # Save the plot as an image
    plt.savefig('static/images/arima_plot.png')

    # Render the arima.html template
    return render_template('arima.html')

# Add Page: Prophet


@app.route('/prophet')
def prophet():
    # Load the data from file
    data = np.load('data.npy')

    # Convert the data to a pandas dataframe
    df = pd.DataFrame(data, columns=['ds', 'y'])

    # Convert the Date column to a datetime object
    df['ds'] = pd.to_datetime(df['ds'])

    # Create a Prophet model
    clf = Prophet()

    # Fit the model on the data
    clf.fit(df)

    # Use the model to predict on the data
    future = clf.make_future_dataframe(periods=365)
    predictions = clf.predict(future)

    # Create a plot
    fig = clf.plot(predictions)

    # Save the plot as an image
    fig.savefig('static/images/prophet_plot.png')

    # Render the prophet.html template
    return render_template('prophet.html')

# Add Page: Prophet Forecast


@app.route('/prophet_forecast')
def prophet_forecast():
    # Load the data from file
    data = np.load('data.npy')

    # Convert the data to a pandas dataframe
    df = pd.DataFrame(data, columns=['ds', 'y'])

    # Convert the Date column to a datetime object
    df['ds'] = pd.to_datetime(df['ds'])

    # Create a Prophet model
    clf = Prophet()

    # Fit the model on the data
    clf.fit(df)

    # Use the model to predict on the data
    future = clf.make_future_dataframe(periods=365)
    predictions = clf.predict(future)

    # Print the predictions
    print(predictions)

    # Render the prophet_forecast.html template
    return render_template('prophet_forecast.html')

# Add Page: Prophet Forecast Plot


@app.route('/prophet_forecast_plot')
def prophet_forecast_plot():
    # Load the data from file
    data = np.load('data.npy')

    # Convert the data to a pandas dataframe
    df = pd.DataFrame(data, columns=['ds', 'y'])

    # Convert the Date column to a datetime object
    df['ds'] = pd.to_datetime(df['ds'])

    # Create a Prophet model
    clf = Prophet()

    # Fit the model on the data
    clf.fit(df)

    # Use the model to predict on the data
    future = clf.make_future_dataframe(periods=365)
    predictions = clf.predict(future)

    # Create a plot
    fig = clf.plot(predictions)

    # Save the plot as an image
    fig.savefig('static/images/prophet_forecast_plot.png')

    # Render the prophet_forecast_plot.html template
    return render_template('prophet_forecast_plot.html')

# Add Page: Prophet Forecast Plot Interactive


@app.route('/prophet_forecast_plot_interactive')
def prophet_forecast_plot_interactive():
    # Load the data from file
    data = np.load('data.npy')

    # Convert the data to a pandas dataframe
    df = pd.DataFrame(data, columns=['ds', 'y'])

    # Convert the Date column to a datetime object
    df['ds'] = pd.to_datetime(df['ds'])

    # Create a Prophet model
    clf = Prophet()

    # Fit the model on the data
    clf.fit(df)

    # Use the model to predict on the data
    future = clf.make_future_dataframe(periods=365)
    predictions = clf.predict(future)

    # Create a plot
    fig = clf.plot(predictions)

    # Save the plot as an image
    fig.savefig('static/images/prophet_forecast_plot_interactive.png')

    # Render the prophet_forecast_plot_interactive.html template
    return render_template('prophet_forecast_plot_interactive.html')

# Add Page: Prophet Forecast Plot Interactive Plotly


@app.route('/prophet_forecast_plot_interactive_plotly')
def prophet_forecast_plot_interactive_plotly():
    # Load the data from file
    data = np.load('data.npy')

    # Convert the data to a pandas dataframe
    df = pd.DataFrame(data, columns=['ds', 'y'])

    # Convert the Date column to a datetime object
    df['ds'] = pd.to_datetime(df['ds'])

    # Create a Prophet model
    clf = Prophet()

    # Fit the model on the data
    clf.fit(df)

    # Use the model to predict on the data
    future = clf.make_future_dataframe(periods=365)
    predictions = clf.predict(future)

    # Create a plot
    fig = clf.plot(predictions)

    # Save the plot as an image
    fig.write_html("templates/prophet_forecast_plot_interactive_plotly.html")

    # Render the prophet_forecast_plot_interactive_plotly.html template
    return render_template('prophet_forecast_plot_interactive_plotly.html')

# Add Page: For Stock Market Prediction using TensorFlow


@app.route('/stock_market_prediction')
def stock_market_prediction():
    # Load the data from file
    data = np.load('data.npy')

    # Convert the data to a pandas dataframe
    df = pd.DataFrame(data, columns=['ds', 'y'])

    # Convert the Date column to a datetime object
    df['ds'] = pd.to_datetime(df['ds'])

    # Create a Prophet model
    clf = Prophet()

    # Fit the model on the data
    clf.fit(df)

    # Use the model to predict on the data
    future = clf.make_future_dataframe(periods=365)
    predictions = clf.predict(future)

    # Create a plot
    fig = clf.plot(predictions)

    # Save the plot as an image
    fig.savefig('static/images/prophet_plot.png')

    # Render the prophet.html template
    return render_template('prophet.html')

# Add Page: For Stock Market Prediction using TensorFlow predict the next day the next 3 days, the next 5 days, the next month, quarter, and year.


@app.route('/stock_market_prediction_predict')
def stock_market_prediction_predict():
    # Load the data from file
    data = np.load('data.npy')

    # Convert the data to a pandas dataframe
    df = pd.DataFrame(data, columns=['ds', 'y'])

    # Convert the Date column to a datetime object
    df['ds'] = pd.to_datetime(df['ds'])

    # Create a Prophet model
    clf = Prophet()

    # Fit the model on the data
    clf.fit(df)

    # Use the model to predict on the data
    future = clf.make_future_dataframe(periods=365)
    predictions = clf.predict(future)

    # Print the predictions
    print(predictions)

    # Render the prophet_forecast.html template
    return render_template('prophet_forecast.html')

# Add Page: For Stock Market Prediction using TensorFlow Plot


@app.route('/stock_market_prediction_plot')
def stock_market_prediction_plot():
    # Load the data from file
    data = np.load('data.npy')

    # Convert the data to a pandas dataframe
    df = pd.DataFrame(data, columns=['ds', 'y'])

    # Convert the Date column to a datetime object
    df['ds'] = pd.to_datetime(df['ds'])

    # Create a Prophet model
    clf = Prophet()

    # Fit the model on the data
    clf.fit(df)

    # Use the model to predict on the data
    future = clf.make_future_dataframe(periods=365)
    predictions = clf.predict(future)

    # Create a plot
    fig = clf.plot(predictions)

    # Save the plot as an image
    fig.savefig('static/images/prophet_plot.png')

    # Render the prophet.html template
    return render_template('prophet.html')


@app.route('/black-scholes-merton', methods=['POST'])
def black_scholes_merton():
    S = float(request.form.get('S'))  # Spot price of the asset
    K = float(request.form.get('K'))  # Strike price of the option
    T = float(request.form.get('T'))  # Time to maturity in years
    r = float(request.form.get('r'))  # Risk-free interest rate
    sigma = float(request.form.get('sigma'))  # Volatility of the asset

    option = Option(S, K, T, r, sigma)
    call_price, put_price = option.calculate_price()

    return jsonify({'call_price': call_price, 'put_price': put_price})


if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5101)
