<?php

// Use the trader extension to calculate technical indicators
use Trader\Momentum;
use Trader\MovingAverage;
use Trader\StandardDeviation;
use Trader\LinearRegression;
use Trader\Stochastics;
use Trader\Greeks;
use Trader\BollingerBands;
use Trader\Ichimoku;
use Trader\PivotPoints;
use Trader\Fibonacci;
use Trader\WilliamsR;
use Trader\RelativeStrengthIndex;
use Trader\AverageTrueRange;
use Trader\CommodityChannelIndex;
use Trader\MovingAverageConvergenceDivergence;
use Trader\ParabolicSAR;
use Trader\ExponentialMovingAverage;
use Trader\SimpleMovingAverage;
use Trader\DoubleExponentialMovingAverage;
use Trader\TripleExponentialMovingAverage;
use Trader\WeightedMovingAverage;
use Trader\HullMovingAverage;
use Trader\KaufmanAdaptiveMovingAverage;
use Trader\ExponentialAdaptiveMovingAverage;
use Trader\IchimokuCloud;
use Trader\IchimokuKijunSen;
use Trader\IchimokuTenkanSen;
use Trader\IchimokuChikouSpan;
use Trader\IchimokuSenkouSpanA;
use Trader\IchimokuSenkouSpanB;
use Trader\IchimokuCloudLag;
use Trader\IchimokuKijunSenLag;
use Trader\IchimokuTenkanSenLag;
use Trader\IchimokuChikouSpanLag;
use Trader\IchimokuSenkouSpanALag;
use Trader\IchimokuSenkouSpanBLag;
use Trader\IchimokuCloudLead;
use Trader\IchimokuKijunSenLead;
use Trader\IchimokuTenkanSenLead;
use Trader\IchimokuChikouSpanLead;
use Trader\IchimokuSenkouSpanALead;
use Trader\IchimokuSenkouSpanBLead;
use Trader\IchimokuCloudCrossover;
use Trader\IchimokuKijunSenCrossover;
use Trader\IchimokuTenkanSenCrossover;
use Trader\IchimokuChikouSpanCrossover;
use Trader\IchimokuSenkouSpanACrossover;
use Trader\IchimokuSenkouSpanBCrossover;
use Trader\IchimokuCloudCrossoverLag;
use Trader\IchimokuKijunSenCrossoverLag;
use Trader\IchimokuTenkanSenCrossoverLag;
use Trader\IchimokuChikouSpanCrossoverLag;
use Trader\IchimokuSenkouSpanACrossoverLag;
use Trader\IchimokuSenkouSpanBCrossoverLag;

// Define the symbols to analyze of 100 Companies
$symbols = array('TSLA', 'AAPL', 'GOOG', 'MSFT', 'AMZN', 'FB', 'NFLX', 'NVDA', 'PYPL', 'GM', 'C', 'GOOGL', 'NASDAQ', 'SPY', 'QQQ', 'XOM', 'JNJ', 'JPM', 'WMT', 'DIS', 'HD', 'INTC', 'VZ', 'BAC', 'T', 'KO', 'PFE', 'MRK', 'ABBV', 'PEP', 'CVX', 'CSCO', 'MCD', 'SNAP', 'TWTR', 'META', 'ADBE', 'CMCSA', 'COST', 'ORCL', 'NVS');

// Include the trader extension
include_once('trader.php');

// get_price function
function get_price($symbol) {
    // Make a request to the Yahoo Finance API
    $response = file_get_contents('https://query1.finance.yahoo.com/v8/finance/chart/' . $symbol);

    // Parse the response as JSON
    $data = json_decode($response, true);

    // Return the current price
    return $data['chart']['result'][0]['meta']['regularMarketPrice'];
}

// Get the current price for each symbol
$prices = array();
foreach ($symbols as $symbol) {
    $prices[$symbol] = get_price($symbol);
}

// Function to calculate the momentum
function momentum($prices, $period) {
    $momentum = 0;
    foreach ($prices as $i => $price) {
        if ($i >= $period) {
            $momentum = $price - $prices[$i - $period];
        }
    }
    return $momentum;
}
// Calculate the momentum for each symbol
$momenta = array();
foreach ($symbols as $symbol) {
    $momenta[$symbol] = momentum($prices[$symbol], 10);
}

// Function to calculate the moving average
function moving_average($prices, $period) {
    $moving_average = 0;
    foreach ($prices as $i => $price) {
        if ($i >= $period) {
            $moving_average += $price;
        }
    }
    return $moving_average / $period;
}

// Calculate the moving average for each symbol
$moving_averages = array();
foreach ($symbols as $symbol) {
    $moving_averages[$symbol] = moving_average($prices[$symbol], 10);
}
// Function to calculate the standard deviation
function standard_deviation($prices, $period) {
    $standard_deviation = 0;
    foreach ($prices as $i => $price) {
        if ($i >= $period) {
            $standard_deviation += pow($price - moving_average($prices, $period), 2);
        }
    }
    return sqrt($standard_deviation / $period);
}
// Calculate the standard deviation for each symbol
$standard_deviations = array();
foreach ($symbols as $symbol) {
    $standard_deviations[$symbol] = standard_deviation($prices[$symbol], 10);
}
// Function to calculate the linear regression
function linear_regression($prices, $period) {
    $linear_regression = 0;
    foreach ($prices as $i => $price) {
        if ($i >= $period) {
            $linear_regression += $price * ($i + 1);
        }
    }
    return $linear_regression / $period;
}
// Calculate the linear regression for each symbol
$linear_regressions = array();
foreach ($symbols as $symbol) {
    $linear_regressions[$symbol] = linear_regression($prices[$symbol], 10);
}
// Function to calculate the stochastics
function stochastic($prices, $period) {

    $stochastics = 0;
    foreach ($prices as $i => $price) {
        if ($i >= $period) {
            $stochastics += $price;
        }
    }
    return $stochastics / $period;
}
// Calculate the stochastics for each symbol
$stochastics = array();
foreach ($symbols as $symbol) {
    $stochastics[$symbol] = stochastic($prices[$symbol], 10, 10, 10);
}
// Function to calculate the Greeks
function greeks($prices, $period) {
    $greeks = array();
    foreach ($prices as $i => $price) {
        if ($i >= $period) {
            $greeks['delta'] = delta($prices[$symbol], 100);
            $greeks['gamma'] = gamma($prices[$symbol], 100);
            $greeks['vega'] = vega($prices[$symbol], 100);
            $greeks['theta'] = theta($prices[$symbol], 100);
            $greeks['rho'] = rho($prices[$symbol], 100);
        }
    }
    return $greeks;
}
// Calculate the Greeks for each symbol
$greeks = array();
foreach ($symbols as $symbol) {
    $greeks[$symbol] = greeks($prices[$symbol], 100);
}

// Print the results
echo 'Symbol | Momentum | Moving Average | Standard Deviation | Linear Regression | Stochastics | Greeks' . PHP_EOL;
foreach ($symbols as $symbol) {
    echo $symbol . ' | ' . $momenta[$symbol] . ' | ' . $moving_averages[$symbol] . ' | ' . $standard_deviations[$symbol] . ' | ' . $linear_regressions[$symbol] . ' | ' . $stochastics[$symbol] . ' | ' . $greeks[$symbol] . PHP_EOL;
}

// End of the program<ctrl63> This code is an example of how to use the trader extension in PHP. The code first includes the trader extension, then defines a function to get the current price of a stock. The code then gets the current price for each of the symbols in the $symbols array, calculates the momentum, moving average, standard deviation, linear regression, stochastics, and Greeks for each symbol, and finally prints the results to the console.


?>
<!DOCTYPE html>
    <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>Trader Extension Example</title>
            <style>
                body {
                    font-family: sans-serif;
                }
                h1 {
                    font-size: 1.5em;
                    margin-bottom: 1em;
                }
                p {
                    margin-bottom: 1em;
                }
                table {
                    border-collapse: collapse;
                    width: 100%;
                }
                th, td {
                    padding: 0.5em;
                    border: 1px solid #ccc;
                }
                th {
                    background-color: #eee;
                }
            </style>

        </head>
        <body>
            <h1>Trader Extension Example</h1>
            <p>This is an example of how to use the trader extension in PHP.</p>
            <!-- Get the current price of each symbol -->
            <?php
            foreach ($symbols as $symbol) {
                echo '<p>Getting the current price of ' . $symbol . '...</p>';
                $prices[$symbol] = get_price($symbol);
            }
            ?>

            <!-- Print the results -->
            <table>
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Momentum</th>
                        <th>Moving Average</th>
                        <th>Standard Deviation</th>
                        <th>Linear Regression</th>
                        <th>Stochastics</th>
                        <th>Greeks</th>
                    </tr>
                </thead>
                <tbody>
                    <?php
                    foreach ($symbols as $symbol) {
                        echo '<tr>';
                        echo '<td>' . $symbol . '</td>';
                        echo '<td>' . $momenta[$symbol] . '</td>';
                        echo '<td>' . $moving_averages[$symbol] . '</td>';
                        echo '<td>' . $standard_deviations[$symbol] . '</td>';
                        echo '<td>' . $linear_regressions[$symbol] . '</td>';
                        echo '<td>' . $stochastics[$symbol] . '</td>';
                        echo '<td>' . $greeks[$symbol] . '</td>';
                        echo '</tr>';
                    }
                    ?>
                </tbody>
            </table>
        </body>
    </html>