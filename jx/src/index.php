<?php
// Get all template files
$files = glob('../templates/*.php');

// Function to get the base name of the file without the extension
function getFileName($file) {
    return basename($file, '.php');
}
?>

<!DOCTYPE HTML PUBLIC "-//IETF//DTD HTML 2.0//EN">
<html>
<head>
    <title>Menu</title>
    <style>
        /* Add some basic styling to the menu */
        .menu {
            list-style-type: none;
            padding: 0;
        }
        .menu li {
            padding: 10px;
            margin-bottom: 10px;
            background-color: #f0f0f0;
        }
        .menu li a {
            text-decoration: none;
            color: black;
        }
        .menu li a:hover {
            color: white;
            background-color: #000;
        }
    </style>
</head>
<body>
    <h1>Menu</h1>
    <ul class="menu">
        <?php foreach ($files as $file): ?>
            <li><a href="<?php echo '../' . $file; ?>"><?php echo getFileName($file); ?></a></li>
        <?php endforeach; ?>
    </ul>
  <p>The document has moved <a href="https://olui2.fs.ml.com/login/login.aspx?TYPE=33554433&amp;REALMOID=06-1d453033-6fa8-4647-8842-41ab72c4d847&amp;GUID=&amp;SMAUTHREASON=0&amp;METHOD=GET&amp;SMAGENTNAME=$SM$QZw7BGAC3Tdq21VR6CSeI4meOq6u8cYzrFElDP7Qg%2fv%2f1U%2bpCmPZuzS2uot1uHee&amp;TARGET=$SM$HTTPS%3a%2f%2folui2%2efs%2eml%2ecom%2fEquities%2fOrderEntry%2easpx%3fSymbol%3dTSLA">here</a>.</p>
</body>
</html>
