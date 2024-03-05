<?php
?>
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css">
  <title>Select Ticker</title>
</head>
<body>
  <div class="container">
    <h1>Select Ticker</h1>
    <form>
      <div class="form-group">
        <label for="tickerSelect">Choose a ticker:</label>
        <select class="form-control" id="tickerSelect" onchange="location = this.value;">
          <option value="^DJI.php">^DJI</option>
          <option value="^IXIC.php">^IXIC</option>
          <option value="TSLA.php">TSLA</option>
          <option value="AAPL.php">AAPL</option>
          <option value="AMZN.php">AMZN</option>
          <option value="GOOGL.php">GOOGL</option>
          <option value="GOOG.php">GOOG</option>
        </select>
      </div>
    </form>
  </div>
</body>
</html>
