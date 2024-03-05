<?php
  $ch = curl_init();

  curl_setopt($ch, CURLOPT_URL, '');
  curl_setopt($ch, CURLOPT_RETURNTRANSFER, 1);

  $result = curl_exec($ch);

  if (curl_errno($ch)) {
    echo 'Error:' . curl_error($ch);
  }

  curl_close($ch);
?>