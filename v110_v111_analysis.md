# v110 vs v111 Comparison: Torso Front Semantic Collapse Investigation

The user suspects that the "torso front semantic collapse" in v111 is caused by changes in the generation input state (pre-clean and edit region) rather than just the text prompt.

## 1. Prompt Dump (v110 vs v111)

Based on the code analysis, here are the literal prompt strings for a hypothetical "white shirt" input.

### v110 literal prompt
**Positive:**
`professional portrait photo of a woman with [style], [color] hair color, clean neckline, balanced framing, photorealistic portrait`

**Negative:**
`[style-specific negative], warm orange cast, yellow brassiness, copper tint, reddish tint, ugly, deformed, blurry, low quality, bad anatomy, distorted face, distorted hair, bald patch, artifacts, watermark, signature, cartoon, anime, illustration, painting, drawing, cropped head, cropped hair, cut off hair, top of head out of frame, tight close-up portrait, clipped hairstyle, earring, earrings, drop earrings, dangling earrings, jewelry, ear accessories, necklace, accessories, piercings, dangling side locks, loose dangling side locks, long dangling side locks, dangling face-framing strands, dangling lower side tails, loose side tendrils touching clothing, side locks touching shoulders or clothing`

---

### v111 literal prompt (Current)
**Positive:**
`professional portrait photo of a woman with [style], [color] hair color, clean white tone, plain unpatterned shirt, soft cotton fabric, same original upper garment, connected shoulder cloth, continuous front garment panel, preserved neckline coverage, clean neckline, balanced framing, photorealistic portrait`

**Negative:**
`[style-specific negative], warm orange cast, yellow brassiness, copper tint, reddish tint, black clothes, navy clothes, blue clothes, unrelated new outfit, dramatically changed clothing style, dramatically changed clothing color, salon cape, salon gown, open neckline, deep v-neck, plunging neckline, exposed chest, armor-like chest panel, bib-like front panel, structured breastplate top, ugly, deformed, blurry, low quality, bad anatomy, distorted face, distorted hair, bald patch, artifacts, watermark, signature, cartoon, anime, illustration, painting, drawing, cropped head, cropped hair, cut off hair, top of head out of frame, tight close-up portrait, clipped hairstyle, earring, earrings, drop earrings, dangling earrings, jewelry, ear accessories, necklace, accessories, piercings, dangling side locks, loose dangling side locks, long dangling side locks, dangling face-framing strands, dangling lower side tails, loose side tendrils touching clothing, side locks touching shoulders or clothing`

---

## 2. Generation Input State Comparison

| Component | v110 State | v111 State | Impact on Semantic Collapse |
| :--- | :--- | :--- | :--- |
| **chest_preclean mask** | Smaller bandwidth (`lane_half` likely ~32), higher threshold (`gray_threshold=154`). Only erases deep black strands. | Expanded bandwidth (`lane_half=48`), lower threshold (`gray_threshold=178`). Captures sideways/lighter strands. | **High.** Erases a larger area of the chest, creating a bigger "hole" that SD must fill. |
| **LaMa before-after** | Preserves most of the garment texture; only erases obvious hair. | Erases potentially valid garment parts if they look like hair or fall within the expanded mask. | **High.** The input to SD has a large inpainted blob in the center chest area. |
| **Generation Input (img_512)** | "Garment-rich": SD sees mostly the original shirt with minimal inpainting. | "Hole-rich": SD sees a large, flat, inpainted area on the torso front. | **Critical.** SD lacks structural context for the chest fabric and relies more on the prompt. |
| **Edit Region (mask_512)** | Tighter around the hair. | Potentially expanded if the pre-clean mask is merged into the generation mask. | **Medium.** Larger edit region gives SD more freedom to "hallucinate" the torso. |

---

## 3. Findings & Next Steps

### Findings:
- The **v111 pre-clean** is much more aggressive in the chest area, leading to a larger "blank canvas" for the torso front.
- The **v111 prompts** add many garment-related keywords ("clean white tone", "plain unpatterned shirt", etc.) and many negative constraints ("armor-like chest panel", "bib-like front panel").
- While the prompts *should* help, the lack of structural cues in the expanded `img_512` (due to aggressive LaMa) might be forcing SD to generate the chest from scratch, leading to the "semantic collapse" (e.g., weird fabric folds or incorrect anatomy).

### Proposed Next Step (Ablation):
1.  **Revert pre-clean parameters** to v110 levels to see if preserving the chest structure fixes the collapse.
2.  **Simplify Garment Prompts**: Fix the prompt to a "plain white smooth front panel" (removing the dynamic hint generation) to see if it stabilizes the output.
