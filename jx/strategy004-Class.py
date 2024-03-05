import os
import time
from plotly.offline import plot


class StockPredictor:
    def __init__(self, symbol):
        self.__symbol = symbol
        self.__data = None
        self.__fig = None
        self.__rmse = None

        def fetch_data(self):
            self.__data = yf.download(
                self.__symbol, start='2020-01-01', end='2022-12-31')

            def preprocess_data(self):
                # Preprocess data
                # ...

                def train_model(self):
                    # Train model
                    # ...

                    def make_predictions(self):
                        # Make predictions
                        # ...

                        def plot_results(self):
                            # Plot results
                            # ...

                            def save_plot(self):
                                if not os.path.exists('static/images'):
                                    os.makedirs(
                                        'static/images')

                                    plot_filename = f'static/images/{self.__symbol}_{int(time.time())}.html'
                                    plot(self.__fig, filename=plot_filename,
                                         auto_open=False)

                                    def get_rmse(self):
                                        return self.__rmse

                                        def predict(self):
                                            self.fetch_data()
                                            self.preprocess_data()
                                            self.train_model()
                                            self.make_predictions()
                                            self.plot_results()
                                            self.save_plot()

                                            @app.route('/predict/<symbol>')
                                            def predict(symbol):
                                                predictor = StockPredictor(
                                                    symbol)
                                                predictor.predict()
                                                rmse = predictor.get_rmse()
                                                return render_template('predict.html', plot_filename=predictor.plot_filename, rmse=rmse)
