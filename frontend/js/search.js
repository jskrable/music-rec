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
		html += '<div>' +
				'<p><b>' + entry.metadata_songs_title + '</b> by ' +
				entry.metadata_songs_artist_name
		if (entry.musicbrainz_songs_year != 0) {
			html += ' (' + entry.musicbrainz_songs_year + ')'
		}
		html += '<button id="' + entry.metadata_songs_song_id + '" title="Add ' +
				entry.metadata_songs_title + '" style="float: right;">Add</button>'
		html += '</p></div>'
	})

	// Push html
	$('#search-results').append(html)

	// jQuery action on button click
	// Expand this to pin chosen songs to div above search bar
	// '#chosen-songs-div'
	$(":button").on('click', (b) => {
	    console.log(b.target.id);  
	});
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