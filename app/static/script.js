function wrapProcessedImage(image) {
    return `<h1>Processed image:</h1>
    <img src="${image}" alt="Generated image" height="100">`
}

function process_image() {
    $('#loading').css('display','inline-block');

    var form_data = new FormData($('#upload-file')[0]);
    $.ajax({
        type: 'POST',
        url: '/processImage',
        data: form_data,
        contentType: false,
        cache: false,
        processData: false,
        success: function(response){
	    $('#loading').css('display','none');

	    var data = jQuery.parseJSON(response)

	    var image = data['generated_image']

	    var generated_image_result = "";
	    generated_image_result += wrapProcessedImage(image);

	    var div = document.getElementById("generated_image")
	    div.innerHTML = generated_image_result;
	},
	error: function(error){
	    console.log(error);
	}
    });
}

$(document).ready(function() {
    $(window).keydown(function(event){
	if(event.keyCode == 13) {
	    event.preventDefault();
	    return false;
	}
    });

    $('#upload-file-btn').click(function() {
        process_image();
    })
})