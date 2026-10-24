# Third-party notices

## Software and scope

The top-level MIT License covers original project source code and original documentation only, as specified in LICENSE_SCOPE.md. Third-party datasets, external notebook definitions, generated-output licensing and embedded fonts are separate. No copied/adapted third-party Python source was identified in the candidate. Python libraries are external runtime dependencies; their implementations and binary distributions are not bundled.

## Algorithm references, not copied-source dependencies

- Jindong Wang, Yiqiang Chen, Shuji Hao, Wenjie Feng and Zhiqi Shen. *Balanced Distribution Adaptation for Transfer Learning*. ICDM 2017, 1129–1134. https://doi.org/10.1109/ICDM.2017.150. Public implementation consulted: https://github.com/jindongwang/transferlearning/blob/master/code/traditional/BDA/BDA.py. Its repository declares MIT, copyright 2018 Jindong Wang. Candidate BDA uses its own low-rank/diagnostic implementation and paper-order initialization; no complete upstream function was identified as copied.
- Dapeng Hu, Mi Luo, Jian Liang and Chuan-Sheng Foo. *Towards Reliable Model Selection for Unsupervised Domain Adaptation: An Empirical Study and A Certified Baseline*. NeurIPS 2024, 37, 135883–135903. https://doi.org/10.52202/079017-4316. EnsV-onehot is a project hard-prediction applicability implementation, not the original probabilistic/NLL setting. No upstream EnsV source is distributed.

## UCI Gas Sensor Array Drift Dataset

Alexander Vergara (2012), *Gas Sensor Array Drift Dataset*. UCI Machine Learning Repository, https://doi.org/10.24432/C5RP6W, source https://archive.ics.uci.edu/dataset/224/gas%2Bsensor%2Barray%2Bdrift%2Bdataset. Official license field CC BY 4.0; historical research-only/noncommercial wording remains unresolved. No raw UCI observations included.

## External long-term e-nose dataset

Julius Wörner, Jonas Eimler and Miriam Pein-Hackelbusch (2025), *Long-Term Drift Behavior of Electronic Nose*, v1. Zenodo, https://doi.org/10.5281/zenodo.15681119. The DOI-specific DataCite rights metadata at https://api.datacite.org/dois/10.5281/zenodo.15681119 declares CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). No raw recordings or reconstructed feature values included. The external benchmark is a project reconstruction/protocol, not an official replacement processed table. See docs/DATA_ATTRIBUTION.md for modification credit.

## External feature-extraction notebook

Supplied feature_extraction.ipynb: copyright 2025 Julius Wörner and Jonas Eimler, author Julius Wörner. Obtained separately by the user; no original function bodies are included here. Users must obtain the notebook from the original Zenodo record. A record-level dataset license does not automatically establish a separate MIT software license for this notebook. No MIT license is asserted for that notebook.

## DejaVu fonts — embedded use in generated files

The six project-generated PDFs include DejaVu Sans subsets. The plots contain no third-party photos, logos or screenshots. DejaVu incorporates Bitstream Vera and Arev material under their own permissive notices; DejaVu changes are public domain. No standalone font files are distributed. The font-related terms below apply to font material embedded in the generated PDFs; they do not license project code or underlying datasets. The notice retains its upstream wording, including public copyright/contact text. These font rights are excluded from the project software license.

```text
Fonts are (c) Bitstream (see below). DejaVu changes are in public domain.
Glyphs imported from Arev fonts are (c) Tavmjong Bah (see below)

Bitstream Vera Fonts Copyright
------------------------------

Copyright (c) 2003 by Bitstream, Inc. All Rights Reserved. Bitstream Vera is
a trademark of Bitstream, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of the fonts accompanying this license ("Fonts") and associated
documentation files (the "Font Software"), to reproduce and distribute the
Font Software, including without limitation the rights to use, copy, merge,
publish, distribute, and/or sell copies of the Font Software, and to permit
persons to whom the Font Software is furnished to do so, subject to the
following conditions:

The above copyright and trademark notices and this permission notice shall
be included in all copies of one or more of the Font Software typefaces.

The Font Software may be modified, altered, or added to, and in particular
the designs of glyphs or characters in the Fonts may be modified and
additional glyphs or characters may be added to the Fonts, only if the fonts
are renamed to names not containing either the words "Bitstream" or the word
"Vera".

This License becomes null and void to the extent applicable to Fonts or Font
Software that has been modified and is distributed under the "Bitstream
Vera" names.

The Font Software may be sold as part of a larger software package but no
copy of one or more of the Font Software typefaces may be sold by itself.

THE FONT SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO ANY WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT OF COPYRIGHT, PATENT,
TRADEMARK, OR OTHER RIGHT. IN NO EVENT SHALL BITSTREAM OR THE GNOME
FOUNDATION BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, INCLUDING
ANY GENERAL, SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF
THE USE OR INABILITY TO USE THE FONT SOFTWARE OR FROM OTHER DEALINGS IN THE
FONT SOFTWARE.

Except as contained in this notice, the names of Gnome, the Gnome
Foundation, and Bitstream Inc., shall not be used in advertising or
otherwise to promote the sale, use or other dealings in this Font Software
without prior written authorization from the Gnome Foundation or Bitstream
Inc., respectively. For further information, contact: fonts at gnome dot
org. 

Arev Fonts Copyright
------------------------------

Copyright (c) 2006 by Tavmjong Bah. All Rights Reserved.

Permission is hereby granted, free of charge, to any person obtaining
a copy of the fonts accompanying this license ("Fonts") and
associated documentation files (the "Font Software"), to reproduce
and distribute the modifications to the Bitstream Vera Font Software,
including without limitation the rights to use, copy, merge, publish,
distribute, and/or sell copies of the Font Software, and to permit
persons to whom the Font Software is furnished to do so, subject to
the following conditions:

The above copyright and trademark notices and this permission notice
shall be included in all copies of one or more of the Font Software
typefaces.

The Font Software may be modified, altered, or added to, and in
particular the designs of glyphs or characters in the Fonts may be
modified and additional glyphs or characters may be added to the
Fonts, only if the fonts are renamed to names not containing either
the words "Tavmjong Bah" or the word "Arev".

This License becomes null and void to the extent applicable to Fonts
or Font Software that has been modified and is distributed under the 
"Tavmjong Bah Arev" names.

The Font Software may be sold as part of a larger software package but
no copy of one or more of the Font Software typefaces may be sold by
itself.

THE FONT SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO ANY WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT
OF COPYRIGHT, PATENT, TRADEMARK, OR OTHER RIGHT. IN NO EVENT SHALL
TAVMJONG BAH BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
INCLUDING ANY GENERAL, SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL
DAMAGES, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF THE USE OR INABILITY TO USE THE FONT SOFTWARE OR FROM
OTHER DEALINGS IN THE FONT SOFTWARE.

Except as contained in this notice, the name of Tavmjong Bah shall not
be used in advertising or otherwise to promote the sale, use or other
dealings in this Font Software without prior written authorization
from Tavmjong Bah. For further information, contact: tavmjong @ free
. fr.

$Id: LICENSE 2133 2007-11-28 02:46:28Z lechimp $
```
