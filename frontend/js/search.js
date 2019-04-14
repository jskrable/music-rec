// search.js

search_terms = []

$(function() {

	loadSongs();


    $('#search-bar').keyup((e) => {
        var value = $('#search-bar').val();
        search_terms.push(value.toLowerCase());
        if (value != "") {
            displaySongs(searchOnType(value));
            }
    });

    // Include enters
    $('#search-terms-in').keypress((e) => {
        if (e.which == 13) {
            $('#search-btn').click();
        }
    });

});



// Pushes song list to page for display
function displaySongs(list) {

	// Clear div
	$('#search-results').empty()

	// Init vars
	html = ''
	// Max 15 songs displayed
	list = list.slice(0,15)

	// Loop thru list and add each song
	list.forEach(function(entry) {
		html += '<li><strong>' + entry.metadata_songs_title + '</strong> by ' +
				entry.metadata_songs_artist_name
		if (entry.musicbrainz_songs_year != 0) {
			html += ' (' + entry.musicbrainz_songs_year + ')'
		}
	})

	// Push html
	$('#search-results').append(html)
}


// Grabs all song lookup metadata and stores in session storage
function loadSongs() {

	// Hit flask API for lookup data
	fetch('http://localhost:5001/lookup')
		.then(function(response) {
    		if (response.ok && response.status == 200) {
    			data = response.json().then(function(data) {
    				// Push to session storage if successful request
    				sessionStorage.setItem('song-lookup', JSON.stringify(data.entity));
    				return data;
    			})
    		} else {
    			console.log(response)
    		}
    	}, function(error) {
    		console.log(error)
    	}
  		)
}


// Filters session storage entity on search terms
function searchOnType(term) {

	// Init filtered list
	filtered = []
	songs = JSON.parse(sessionStorage.getItem('song-lookup'));

	// Perform filter searchs
	filtered = songs.filter(function(song) {
		match = Object.values(song).toString().toLowerCase()
				.indexOf(term.toLowerCase()) > 0 ?
					true :
					false
		return match
		});

	return filtered
}