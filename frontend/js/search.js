// search.js

search_terms = []
selected_songs = []

$(function() {

	loadSongs();


    $('#search-bar').keyup((e) => {
        var value = $('#search-bar').val();
        search_terms.push(value.toLowerCase());
        if (value != "") {
            displaySongs(searchOnType(value));
            /*setTimeout(function(){
				displaySongs(searchOnType(value));
			}, 2500);*/
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
		html += '<button id="add-' + entry.metadata_songs_song_id + 
				'" value="' + entry.metadata_songs_title + '" title="Add ' +
				entry.metadata_songs_title + '" style="float: right;">Add</button>'
		html += '</p></div>'
	})

	// Push html
	$('#search-results').append(html)

	// jQuery action on button click
	// Expand this to pin chosen songs to div above search bar
	// '#chosen-songs-div'
	$(":button").on('click', (b) => {

		if (b.target.id.split('-')[0] == 'add') {

			if (selected_songs.length >= 3) {
				$('#song-limit-alert-div').show()
				setTimeout(function(){
					$('#song-limit-alert-div').fadeOut("slow");
				}, 1500);

			} else {
			    console.log(b.target.id);
			    console.log(b.target.value);
			    $('#predict-div').show();
		        key = b.target.id.split('-')[1];
		        title = b.target.value;
		        selected_songs.push(key);

		        html = '<div class="four columns" id="' + key + 
		        	   '"><strong>' + title + 
		        	   '</strong> <a href="#" id="drop-' + key + 
		        	   '" onclick="closeSong(this)">Remove?</a>'
		        	   + '</div>'

		        $('#chosen-songs-div').append(html);
		    }
	    }

	});
}


// Pushes song list to page for display
function displayRecs(data) {

	//console.log(data)

	// Init vars
	html = ''
	list = data.entity.recommendations

	// Loop thru list and add each song
	list.forEach(function(entry) {
        html += '<div>'
        html += '<p><b>' + entry.metadata_songs_title + '</b> by ' +
                entry.metadata_songs_artist_name + ' '
        if (entry.musicbrainz_songs_year != 0) {
            html += '(' + entry.musicbrainz_songs_year + ') '
        }
        html += '<a href="https://www.google.com/search?q='
        query = [entry.metadata_songs_title, entry.metadata_songs_artist_name]
                    .join(' ').toLowerCase().split(' ').join('+')
        html += query + '" target="_blank">Listen</a>'
        html += '</p></div>'
	})

	// Push html
	$('#recs-div').append(html)
	$('#search-container').hide()
	$('#recs-container').show()


}


function restart() {

	$('#recs-div').empty()
	$('#search-container').show()
	$('#recs-container').hide()	
	clearSelected()
	$('#search-bar').val('')
}

// Grabs all song lookup metadata and stores in session storage
function loadSongs() {

	console.log('Hitting lookup API...')

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


function closeSong(song) {

	console.log(song.id)
    key = song.id.split('-')[1];
    var index = selected_songs.indexOf(key);
    if (index > -1) {
        selected_songs.splice(index, 1);
        $('#'+key+'').remove()
    }
    if (selected_songs.length < 1) {
        $('#predict-div').hide();
    }
}


function getRecs() {

	url = 'http://localhost:5001/recommend?songs=' + selected_songs

	console.log('Hitting recommendations API at ' + url)
	// Hit flask API to get recommendations from model
	fetch(url)
		.then(function(response) {
    		if (response.ok && response.status == 200) {
    			data = response.json().then(function(data) {
    				displayRecs(data);
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


function clearSelected() {

	$('#chosen-songs-div').empty()
	$('#predict-div').hide()
	selected_songs = []

}