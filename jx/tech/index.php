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
    <?php
      // Include navbar.php and check the previous direcotry if it is not in the current or tech direcotry.
      if (file_exists('../navbar.php')) {
        include '../navbar.php';
      } elseif (file_exists('navbar.php')) {
        include 'navbar.php';
      } else {
        include 'tech/navbar.php';
      }

      // Insert a responsive jumbotron with a welcome message.
      // include 'jumbotron.php';

      // Insert a responsive table that shows the latest stock data for the selected ticker.
      // include 'table.php';
    ?>
    <h1>Select Ticker</h1>
    <form>
      <div class="form-group">
        <label for="tickerSelect">Choose a ticker:</label>
        <select class="form-control" id="tickerSelect" onchange="location = this.value + '.php';">
          <option value="^DJI.php">^DJI</option>
          <option value="^IXIC.php">^IXIC</option>
          <option value="TSLA.php">TSLA</option>
          <option value="AAPL.php">AAPL</option>
          <option value="AMZN.php">AMZN</option>
          <option value="GOOGL.php">GOOGL</option>
          <option value="GOOG.php">GOOG</option>
        </select>
			<input type="submit" value="Submit">
      </div>
    </form>
  </div>
</body>
</html>
