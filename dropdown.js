<html>
<head>
<title>dropdown menu using select tab</title>
</head>
<script>
function favTutorial() {
var mylist = document.getElementById("myList");
document.getElementById("favourite").value = mylist.options[mylist.selectedIndex].text;
}
</script>

<body>
<form>
<b> Select you favourite tutorial site using dropdown list </b>
<select id = "myList" onchange = "favTutorial()" >
<option> ---Choose tutorial--- </option>
<option> w3schools </option>
<option> Javatpoint </option>
<option> tutorialspoint </option>
<option> geeksforgeeks </option>
</select>
<p> Your selected tutorial site is: 
<input type = "text" id = "favourite" size = "20" </p>
</form>
</body>
</html>
