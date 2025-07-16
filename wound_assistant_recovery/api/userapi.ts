import type { IImage } from "../types/imageType";
import type {
  ILogInUser,
  ILoginResponse,
  IRegisterUser,
  IUser,
} from "../types/usertype";
import strapi from "./strapi";

export const UserHandler = {
  async logIn(user: ILogInUser): Promise<ILoginResponse> {
    try {
      const { data } = await strapi.post("/auth/local", user);
      return data;
    } catch (error) {
      console.log(JSON.stringify(error));
      throw error;
    }
  },

  async register(user: IRegisterUser): Promise<ILoginResponse> {
    const image = user.profile ? await this.uploadImage(user.profile) : null;
    console.log(user);
    const { data } = await strapi.post("/auth/local/register", {
      ...user,
      profile: image?.id || null,
    });

    return data;
  },

  async getLoginUser(userId: number): Promise<IUser> {
    const { data } = await strapi.get(`/users/${userId}`);
    return data;
  },

  async logout() {
    await strapi.post("/logout");
  },

  async uploadImage(img: File): Promise<IImage> {
    const formData = new FormData();
    formData.append("files", img);
    const { data } = await strapi.post<IImage[]>("/upload", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return data[0];
  },
};
