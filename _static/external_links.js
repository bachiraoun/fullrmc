/*
 * fullrmc_theme/static/external_links.js
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * Forces every hyperlink that does not point back into this documentation
 * to open in a new tab. This covers two very different kinds of links in
 * one generic rule instead of two:
 *
 *   1. Docutils-generated external links (RST `text <url>`_ syntax) -- these
 *      always carry a "reference external" class, but relying on the class
 *      alone would still miss...
 *   2. Hand-written links baked directly into the theme templates (e.g. the
 *      "Cloud Services" nav item, the Sphinx credit in the footer) -- these
 *      never go through docutils, so they never get that class.
 *
 * Rather than hardcoding URL substrings or chasing every new link added in
 * the future, this compares each link's resolved hostname against the
 * current page's hostname. Anything that resolves to a different host is,
 * by definition, "not this doc" and gets target="_blank". Internal
 * cross-page navigation, table-of-contents links, and #anchors all resolve
 * to the same hostname and are left alone.
 */
(function () {
    "use strict";

    function isExternal(link) {
        // Only genuine http(s) navigations count -- leave mailto:, tel:,
        // and javascript: pseudo-links untouched.
        if (link.protocol !== "http:" && link.protocol !== "https:") {
            return false;
        }
        return link.hostname !== window.location.hostname;
    }

    function applyNewTabLinks() {
        var links = document.querySelectorAll("a[href]");
        for (var i = 0; i < links.length; i++) {
            var link = links[i];
            if (isExternal(link)) {
                link.setAttribute("target", "_blank");
                link.setAttribute("rel", "noopener noreferrer");
            }
        }
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", applyNewTabLinks);
    } else {
        applyNewTabLinks();
    }
})();
