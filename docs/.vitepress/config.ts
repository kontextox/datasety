import { defineConfig } from "vitepress";

export default defineConfig({
  title: "datasety",
  description:
    "CLI tool for dataset preparation: resize, caption, align, shuffle, synthetic, mask, degrade, character, sweep, workflow, train",
  base: "/datasety/",
  head: [
    ["meta", { name: "theme-color", content: "#5b7ee5" }],
    ["meta", { name: "og:type", content: "website" }],
    ["meta", { name: "og:title", content: "datasety" }],
    [
      "meta",
      {
        name: "og:description",
        content: "Creating a AI dataset for LoRA/Fine-tuning (deep learning)",
      },
    ],
    ["link", { rel: "icon", href: "/datasety/favicon.ico" }],
  ],

  themeConfig: {
    logo: undefined,

    nav: [
      { text: "Guide", link: "/getting-started" },
      {
        text: "Commands",
        items: [
          { text: "resize", link: "/commands/resize" },
          { text: "caption", link: "/commands/caption" },
          { text: "align", link: "/commands/align" },
          { text: "shuffle", link: "/commands/shuffle" },
          { text: "synthetic", link: "/commands/synthetic" },
          { text: "mask", link: "/commands/mask" },
          { text: "degrade", link: "/commands/degrade" },
          { text: "character", link: "/commands/character" },
          { text: "sweep", link: "/commands/sweep" },
          { text: "workflow", link: "/commands/workflow" },
          { text: "train", link: "/commands/train" },
        ],
      },
      { text: "Workflows", link: "/workflows" },
    ],

    sidebar: {
      "/": [
        {
          text: "Guide",
          items: [
            { text: "Getting Started", link: "/getting-started" },
            { text: "Workflows", link: "/workflows" },
          ],
        },
        {
          text: "Image Processing",
          collapsed: false,
          items: [
            { text: "resize", link: "/commands/resize" },
            { text: "caption", link: "/commands/caption" },
            { text: "align", link: "/commands/align" },
            { text: "mask", link: "/commands/mask" },
            { text: "degrade", link: "/commands/degrade" },
          ],
        },
        {
          text: "Generation",
          collapsed: false,
          items: [
            { text: "synthetic", link: "/commands/synthetic" },
            { text: "character", link: "/commands/character" },
            { text: "shuffle", link: "/commands/shuffle" },
          ],
        },
        {
          text: "Automation",
          collapsed: false,
          items: [
            { text: "sweep", link: "/commands/sweep" },
            { text: "workflow", link: "/commands/workflow" },
          ],
        },
        {
          text: "Training",
          collapsed: false,
          items: [
            { text: "train", link: "/commands/train" },
          ],
        },
      ],
    },

    socialLinks: [
      { icon: "github", link: "https://github.com/kontextox/datasety" },
    ],

    footer: {
      message: "Released under the MIT License.",
      copyright: "Copyright KontextoX",
    },

    search: {
      provider: "local",
    },

    editLink: {
      pattern: "https://github.com/kontextox/datasety/edit/main/docs/:path",
      text: "Edit this page on GitHub",
    },
  },
});
