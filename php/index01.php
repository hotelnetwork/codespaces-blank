<?php

// Include the trader extension
include_once('trader.php');

// Define the symbols to analyze
$symbols = array('AAPL', 'GOOG', 'MSFT');

// Define a function to get the current price for a symbol
function get_price($symbol) {
    // Get the current price from Yahoo Finance
    $url = 'https://finance.yahoo.com/quote/' . $symbol . '?p=' . $symbol;
    $html = file_get_contents($url);

    // Parse the HTML to get the current price
    $pattern = '/<span class="Trsdu(C\(.*?\))?">(.*?)<\/span>/';
    preg_match($pattern, $html, $matches);

    // Return the current price
    return $matches[2];
}
// Get the current price for each symbol
$prices = array();
foreach ($symbols as $symbol) {
    $prices[$symbol] = get_price($symbol);
}

// Calculate the momentum for each symbol
$momenta = array();
foreach ($symbols as $symbol) {
    $momenta[$symbol] = trader_momentum($prices[$symbol], 10);
}

// Calculate the moving average for each symbol
$moving_averages = array();
foreach ($symbols as $symbol) {
    $moving_averages[$symbol] = trader_moving_average($prices[$symbol], 10);
}

// Calculate the standard deviation for each symbol
$standard_deviations = array();
foreach ($symbols as $symbol) {
    $standard_deviations[$symbol] = trader_standard_deviation($prices[$symbol], 10);
}

// Calculate the linear regression for each symbol
$linear_regressions = array();
foreach ($symbols as $symbol) {
    $linear_regressions[$symbol] = trader_linear_regression($prices[$symbol], 10);
}

// Calculate the stochastics for each symbol
$stochastics = array();
foreach ($symbols as $symbol) {
    $stochastics[$symbol] = trader_stochastic($prices[$symbol], 10, 10, 10);
}

// Calculate the Greeks for each symbol
$greeks = array();
foreach ($symbols as $symbol) {
    $greeks[$symbol] = trader_greeks($prices[$symbol], 100);
}

// Print the results
echo 'Symbol | Momentum | Moving Average | Standard Deviation | Linear Regression | Stochastics | Greeks' . PHP_EOL;
foreach ($symbols as $symbol) {
    echo $symbol . ' | ' . $momenta[$symbol] . ' | ' . $moving_averages[$symbol] . ' | ' . $standard_deviations[$symbol] . ' | ' . $linear_regressions[$symbol] . ' | ' . $stochastics[$symbol] . ' | ' . $greeks[$symbol] . PHP_EOL;
}

?>
